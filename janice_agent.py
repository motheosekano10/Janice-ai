import gradio as gr
import os
import yaml
import speech_recognition as sr
from gtts import gTTS
import tempfile
import re

# 1. Import Zeline Agent and Tools
from zeline import Agent
try:
    from sa_tenders import sa_tenders
except ImportError:
    sa_tenders = None

# 2. Define the Tender Discovery Tool
def find_tenders(keyword: str, start_date: str = "2026-01-01"):
    """
    Finds government tenders on the South African eTenders portal.
    Args:
        keyword: The search term (e.g., "IT services", "construction").
        start_date: The earliest date for tenders (YYYY-MM-DD).
    """
    if not sa_tenders:
        return "Tender tool is currently unavailable. Please check library installation."
        
    print(f"🔧 Using tool: find_tenders(keyword='{keyword}', start_date='{start_date}')")
    try:
        source = sa_tenders(
            page_size=50,
            start_date=start_date,
            end_date="2026-12-31"
        )
        clean_data = source.cleaned_dataframe
        filtered_tenders = clean_data[clean_data['description'].str.contains(keyword, case=False, na=False)]
        
        if filtered_tenders.empty:
            return f"No tenders found for '{keyword}' after {start_date}."
        
        summary = "Found the following tenders:\n"
        for index, row in filtered_tenders.head(3).iterrows():
            summary += f"- {row['description']} (Closing: {row['closing_date']})\n"
        return summary
    except Exception as e:
        return f"Error fetching tenders: {str(e)}"

# 3. Main Agentic Function
def ask_janice(message, audio_path, history):
    # Handle Audio Input
    if audio_path:
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                message = recognizer.recognize_google(audio_data)
        except Exception as e:
            history.append({"role": "assistant", "content": f"❌ Audio error: {e}"})
            return history, None

    if not message:
        history.append({"role": "assistant", "content": "Please provide a question or command."})
        return history, None

    # Initialize the Agent (Using OpenRouter as the backend)
    # Zeline uses LiteLLM, which supports OpenRouter natively via the model string
    agent = Agent(
        model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        tools=[find_tenders]
    )

    # Run the agent
    try:
        # The agent will decide if it needs to use the find_tenders tool
        response = agent.run(message)
        final_text = str(response)

        # Generate Audio (TTS)
        audio_filepath = None
        try:
            tts = gTTS(text=final_text[:1500], lang='en') # Limit to 1500 chars for speed
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            audio_filepath = temp_file.name
        except Exception as e:
            print(f"TTS Error: {e}")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": final_text})
        
        # Return audio_filepath. The UI will NOT autoplay it.
        return history, audio_filepath

    except Exception as e:
        error_msg = f"**Agent Error:** {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, None

# 4. Gradio Interface
with gr.Blocks(title="Janice AI Agent") as demo:
    gr.Markdown("# 🧠 Janice: Your Autonomous Assistant")
    gr.Markdown("Ask me to find tenders, research the web, or automate tasks.")
    
    chatbot = gr.Chatbot(label="Janice's Activity Log", height=500)
    
    with gr.Row():
        with gr.Column(scale=4):
            text_in = gr.Textbox(show_label=False, placeholder="Type your command here...", container=False)
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Or speak")
            
    with gr.Row():
        submit_btn = gr.Button("Execute", variant="primary")
        
    with gr.Row():
        # Notice: autoplay is NOT here. You must click play manually.
        audio_out = gr.Audio(label="Listen to Verdict", interactive=True)
        
    submit_btn.click(
        fn=ask_janice, 
        inputs=[text_in, audio_in, chatbot], 
        outputs=[chatbot, audio_out]
    )
    
    # Clear inputs after submission
    submit_btn.click(lambda: (None, None), outputs=[text_in, audio_in])

# 5. Launch
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
