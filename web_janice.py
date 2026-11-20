import gradio as gr
import subprocess
import os
import yaml
import re
import speech_recognition as sr
from gtts import gTTS
import tempfile
from openai import OpenAI

# 1. Setup the Config (For the debate tool)
config_dir = os.path.expanduser("~/.parliament")
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, "config.yaml")

config_data = {
    "parliament": {
        "members": [
            {"name": "Llama-3.3", "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"},
            {"name": "Nemotron-Lightning", "provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free"},
            {"name": "Ling-Flash", "provider": "openrouter", "model": "inclusionai/ling-3.2-flash:free"}
        ],
        "speaker": {
            "name": "Lead-Architect",
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free"
        }
    }
}

with open(config_path, "w") as f:
    yaml.dump(config_data, f)

# 2. Setup Normal Assistant Client (OpenRouter)
# Make sure OPENROUTER_API_KEY is set in Render
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

# 3. Helper to clean terminal output
def clean_terminal_output(raw_text):
    cleaned = re.sub(r'[│╭╮╰╯─]', '', raw_text)
    if "Parliament Verdict" in cleaned:
        verdict_section = cleaned.split("Parliament Verdict")[-1]
        verdict_section = re.sub(r'\n\s*\n', '\n\n', verdict_section)
        return "### 🏛️ Janice's Final Verdict\n" + verdict_section.strip()
    return cleaned

# 4. The Smart Assistant Function
def ask_janice(message, audio_path, history):
    # Handle Audio Input
    if audio_path:
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                message = recognizer.recognize_google(audio_data)
        except Exception as e:
            history.append({"role": "user", "content": "🎤 [Audio Input]"})
            history.append({"role": "assistant", "content": f"❌ Error transcribing audio: {e}"})
            return history, None

    if not message:
        history.append({"role": "user", "content": "..."})
        history.append({"role": "assistant", "content": "Please provide a message."})
        return history, None

    # Build conversation history for standard chat
    messages = [{"role": "system", "content": "You are Janice, a helpful, concise, and intelligent personal assistant. Answer normal queries directly. If the user wants a heavy analysis, tell them to use the word 'debate'."}]
    for msg in history[-6:]: # Keep last 3 exchanges for context
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    # --- SMART ROUTER ---
    # If the user asks for a debate/analysis, trigger the Parliament
    trigger_words = ["debate", "pros and cons", "analyze", "compare options", "panel"]
    if any(word in message.lower() for word in trigger_words):
        try:
            env = os.environ.copy()
            env["TERM"] = "dumb"
            
            # Run Parliament
            result = subprocess.run(
                ["parliament", "ask", message],
                capture_output=True, text=True, env=env, timeout=300
            )
            raw_output = result.stdout + "\n" + result.stderr
            final_response = clean_terminal_output(raw_output)
            
            if "empty response" in final_response.lower() or "error" in final_response.lower():
                final_response = "I tried to run a debate, but hit an error. Here is the log:\n\n" + raw_output
                
        except Exception as e:
            final_response = f"**Debate Error:** {str(e)}"
    else:
        # Run Standard Chat
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free", # Fast, smart free model
                messages=messages,
                timeout=60
            )
            final_response = response.choices[0].message.content
        except Exception as e:
            final_response = f"**Chat Error:** {str(e)}"

    # Generate Audio (TTS)
    audio_filepath = None
    try:
        voice_text = final_response[:1000] # Limit TTS length
        tts = gTTS(text=voice_text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        audio_filepath = temp_file.name
    except Exception as e:
        print(f"TTS Error: {e}")

    # Update Chat History
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_response})
    
    return history, audio_filepath

# 5. Gradio Interface
with gr.Blocks(title="Janice AI") as demo:
    gr.Markdown("# 🧠 Janice: Your Personal Assistant")
    gr.Markdown("Type or speak normally. To trigger the multi-agent panel, use words like **'debate'**, **'analyze'**, or **'pros and cons'**.")
    
    chatbot = gr.Chatbot(label="Janice", height=500)
    
    with gr.Row():
        with gr.Column(scale=4):
            text_in = gr.Textbox(show_label=False, placeholder="Say something...", container=False)
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Or speak")
            
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        
    with gr.Row():
        audio_out = gr.Audio(label="Listen to Response", interactive=True)
        
    submit_btn.click(
        fn=ask_janice, 
        inputs=[text_in, audio_in, chatbot], 
        outputs=[chatbot, audio_out]
    )
    
    submit_btn.click(lambda: (None, None), outputs=[text_in, audio_in])

# 6. Launch
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
