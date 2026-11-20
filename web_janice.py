import gradio as gr
import subprocess
import os
import yaml
import re

# 1. Set up the Debate Config (Swapped Gemma for Llama-3.3 to avoid rate limits)
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
            "name": "Judge-Nemotron-Ultra",
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free"
        }
    }
}

with open(config_path, "w") as f:
    yaml.dump(config_data, f)

# 2. Helper function to clean the ugly terminal boxes
def clean_terminal_output(raw_text):
    # Remove the ASCII box characters
    cleaned = re.sub(r'[│╭╮╰╯─]', '', raw_text)
    
    # Grab only the final verdict section to make it readable
    if "Parliament Verdict" in cleaned:
        parts = cleaned.split("Parliament Verdict")
        verdict_section = parts[-1]
        
        # Clean up extra whitespace
        verdict_section = re.sub(r'\n\s*\n', '\n\n', verdict_section)
        verdict_section = re.sub(r'\n\s+', '\n', verdict_section)
        
        return "### 🏛️ Janice's Final Verdict\n" + verdict_section.strip()
    
    return cleaned

# 3. The Debate Function (with Follow-up History support)
def ask_janice(message, history):
    try:
        env = os.environ.copy()
        
        # Build context from previous messages (up to the last 2 exchanges)
        context = ""
        if history:
            context = "Previous conversation context:\n"
            for user_msg, ai_msg in history[-2:]:
                context += f"User: {user_msg}\n"
                # Just take the first 500 chars of previous AI responses to save tokens
                context += f"AI: {str(ai_msg)[:500]}...\n\n"
        
        # Combine context with the new question
        full_prompt = f"{context}New Question: {message}" if context else message

        # Run the parliament
        result = subprocess.run(
            ["parliament", "ask", full_prompt],
            capture_output=True,
            text=True,
            env=env,
            timeout=300
        )
        
        if result.returncode == 0:
            # Clean the output before showing it on the webpage
            return clean_terminal_output(result.stdout)
        else:
            return f"**Parliament Error:**\n{result.stderr}\n\n**Output:**\n{result.stdout}"
            
    except Exception as e:
        return f"**Unexpected Error:** {str(e)}"

# 4. Build the Chat Interface
with gr.Blocks(title="Janice AI") as demo:
    gr.Markdown("# 🧠 Janice: Multi-Agent Debate Panel")
    gr.Markdown("Ask a question. Janice will have her models debate it, and you can ask follow-ups!")
    
    # ChatInterface automatically handles the history and the text box!
    chat = gr.ChatInterface(
        fn=ask_janice,
        title="",
        description="",
        textbox=gr.Textbox(placeholder="Should I expand my business?", container=False, scale=7),
    )

# 5. Launch
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
