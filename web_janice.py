import gradio as gr
import subprocess
import os
import yaml
import re
import speech_recognition as sr
from gtts import gTTS
import tempfile
import json
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# 1. Setup Parliament Config
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

# 2. Setup OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

# 3. Define the Tools (Automation)
def search_tenders(keyword: str):
    """Scrapes the SA eTenders website for a keyword."""
    print(f"🔧 Running tool: search_tenders('{keyword}')")
    try:
        url = f"https://www.etenders.gov.za/Home/opportunities?search={keyword}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            tenders = soup.find_all('h3', limit=5) 
            if not tenders:
                return f"Found the page for '{keyword}', but could not extract specific tender titles. You may need to visit manually."
            
            results = [t.get_text(strip=True) for t in tenders]
            return "Latest tenders found:\n" + "\n".join([f"- {r}" for r in results])
        else:
            return f"Failed to fetch tenders. Status code: {response.status_code}"
    except Exception as e:
        return f"Tool Error: {str(e)}"

# Tool Schema for OpenRouter (OpenAI format)
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_tenders",
            "description": "Search the South African eTenders website for government contracts based on a keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The industry or product to search for (e.g., 'IT services', 'construction', 'catering')."
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]

# 4. Helper to clean terminal output
def clean_terminal_output(raw_text):
    cleaned = re.sub(r'[│╭╮╰╯─]', '', raw_text)
    if "Parliament Verdict" in cleaned:
        verdict_section = cleaned.split("Parliament Verdict")[-1]
        verdict_section = re.sub(r'\n\s*\n', '\n\n', verdict_section)
        return "### 🏛️ Janice's Final Verdict\n" + verdict_section.strip()
    return cleaned

# 5. The Smart Assistant Function with Tool Calling
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

    # Build conversation history
    messages = [{"role": "system", "content": "You are Janice, a helpful personal assistant. You can answer questions and use tools to find tenders. If the user wants a heavy analysis, tell them to use the word 'debate'."}]
    for msg in history[-6:]: 
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    # --- SMART ROUTER & TOOL CALLING ---
    trigger_words = ["debate", "pros and cons", "analyze", "compare options", "panel"]
    
    if any(word in message.lower() for word in trigger_words):
        # Run Parliament (Debate Mode)
        try:
            env = os.environ.copy()
            env["TERM"] = "dumb"
            result = subprocess.run(["parliament", "ask", message], capture_output=True, text=True, env=env, timeout=300)
            raw_output = result.stdout + "\n" + result.stderr
            final_response = clean_terminal_output(raw_output)
            if "empty response" in final_response.lower() or "error" in final_response.lower():
                final_response = "I tried to run a debate, but hit an error. Here is the log:\n\n" + raw_output
        except Exception as e:
            final_response = f"**Debate Error:** {str(e)}"
    else:
        # Run Standard Chat with Tool Calling
        try:
            # First call: Ask the model. It might respond directly or ask to use a tool.
            # 🔥 CHANGED MODEL HERE TO A RELIABLE FREE ALTERNATIVE
            response = client.chat.completions.create(
                model="nvidia/nemotron-3.5-lightning:free", 
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                timeout=60
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Check if the model wants to use a tool
            if tool_calls:
                messages.append(response_message)
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "search_tenders":
                        function_response = search_tenders(keyword=function_args.get("keyword"))
                        
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        })
                
                # Second call: Send the tool results back so the model can formulate the final answer
                second_response = client.chat.completions.create(
                    model="nvidia/nemotron-3.5-lightning:free",
                    messages=messages,
                    timeout=60
                )
                final_response = second_response.choices[0].message.content
            else:
                # No tool needed, just use the direct response
                final_response = response_message.content

        except Exception as e:
            final_response = f"**Chat Error:** {str(e)}"

    # Generate Audio (TTS)
    audio_filepath = None
    try:
        tts = gTTS(text=final_response[:1000], lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        audio_filepath = temp_file.name
    except Exception as e:
        print(f"TTS Error: {e}")

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_response})
    return history, audio_filepath

# 6. Gradio Interface
with gr.Blocks(title="Janice AI") as demo:
    gr.Markdown("# 🧠 Janice: Your Personal Assistant")
    gr.Markdown("Ask normally, or tell me to **'find tenders for [keyword]'**. Use **'debate'** to start a panel discussion.")
    
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
        
    submit_btn.click(fn=ask_janice, inputs=[text_in, audio_in, chatbot], outputs=[chatbot, audio_out])
    submit_btn.click(lambda: (None, None), outputs=[text_in, audio_in])

# 7. Launch
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
