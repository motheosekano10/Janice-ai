import gradio as gr
import subprocess
import os
import yaml
import re
import speech_recognition as sr
from gtts import gTTS
import tempfile

# 1. Setup Configuration (Dedicated Business Panel & Dev Team)
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

# 2. Clean terminal ASCII formatting
def clean_terminal_output(raw_text):
    cleaned = re.sub(r'[│╭╮╰╯─]', '', raw_text)
    if "Parliament Verdict" in cleaned:
        verdict_section = cleaned.split("Parliament Verdict")[-1]
        verdict_section = re.sub(r'\n\s*\n', '\n\n', verdict_section)
        return "### 🏛️ Janice's Final Verdict\n" + verdict_section.strip()
    return cleaned

# 3. Main Function with Panel Routing
def ask_janice(message, audio_path, history):
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
        history.append({"role": "assistant", "content": "Please provide a question."})
        return history, None

    # Build Context from recent history
    context = ""
    if history:
        context = "Previous conversation:\n"
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context += f"{role.capitalize()}: {str(content)[:300]}...\n"

    full_prompt = f"{context}New Question: {message}" if context else message

    # Route based on prompt keywords
    keywords = ["code", "python", "frontend", "backend", "app", "web", "api", "security", "vulnerability", "database", "server"]
    is_dev_panel = any(word in message.lower() for word in keywords)
    panel_name = "🛠️ Dev Team" if is_dev_panel else "📊 Business Panel"

    try:
        env = os.environ.copy()
        env["TERM"] = "dumb"
        
        result = subprocess.run(
            ["parliament", "ask", full_prompt],
            capture_output=True, text=True, env=env, timeout=300
        )
        
        raw_output = result.stdout + "\n" + result.stderr
        clean_output = clean_terminal_output(raw_output)
        
        if not clean_output.strip() or "Parliament Verdict" not in raw_output and len(clean_output) < 50:
            clean_output = f"⚠️ The debate engine returned an empty response. Raw log:\n\n```text\n{raw_output}\n```"
            
    except Exception as e:
        clean_output = f"**Unexpected Error:** {str(e)}"

    # Generate TTS Audio
    audio_filepath = None
    try:
        voice_text = clean_output.split("Recommendation")[-1].strip() if "Recommendation" in clean_output else clean_output
        if not voice_text:
            voice_text = "The debate is complete."
        tts = gTTS(text=voice_text[:1000], lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        audio_filepath = temp_file.name
    except Exception as e:
        print(f"TTS Error: {e}")

    history.append({"role": "user", "content": f"[{panel_name}] {message}"})
    history.append({"role": "assistant", "content": clean_output})
    
    return history, audio_filepath

# 4. Custom Interface
with gr.Blocks(title="Janice AI") as demo:
    gr.Markdown("# 🧠 Janice: Multi-Agent Debate Panel")
    gr.Markdown("Type or speak your question. Janice auto-routes to the **Business Panel** or **Dev Team**.")
    
    chatbot = gr.Chatbot(label="Janice's Deliberations", height=500)
    
    with gr.Row():
        with gr.Column(scale=4):
            text_in = gr.Textbox(show_label=False, placeholder="Type your question here...", container=False)
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Or speak")
            
    with gr.Row():
        submit_btn = gr.Button("Start Debate", variant="primary")
        
    with gr.Row():
        audio_out = gr.Audio(label="Listen to Verdict", autoplay=True)
        
    submit_btn.click(
        fn=ask_janice, 
        inputs=[text_in, audio_in, chatbot], 
        outputs=[chatbot, audio_out]
    )
    
    submit_btn.click(lambda: (None, None), outputs=[text_in, audio_in])

# 5. Launch
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
