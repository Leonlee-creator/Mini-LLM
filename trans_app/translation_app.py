import os
import torch
import gradio as gr
from transformers import pipeline
from pathlib import Path
import socket

# ======================
# 🎨 GOLDEN THEME DESIGN
# ======================
TAD_THEME = {
    "primary": "#FFD700",  # Gold
    "secondary": "#009739",  # Emerald
    "accent": "#5E3023",  # Rich brown
    "text": "#FFFFFF",  # White
    "panel": "#1E1E1E"  # Dark panels
}

CUSTOM_CSS = f"""
.gradio-container {{
    background: linear-gradient(135deg, {TAD_THEME['secondary']} 0%, {TAD_THEME['primary']} 100%);
    max-width: 800px !important;
    border-radius: 20px !important;
    border: 3px solid {TAD_THEME['primary']} !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
}}

.header {{
    text-align: center;
    background: rgba(30, 30, 30, 0.85);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    border: 2px solid {TAD_THEME['primary']};
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}}

.header h1 {{
    background: linear-gradient(90deg, {TAD_THEME['primary']}, {TAD_THEME['secondary']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: -0.5px;
}}

.translation-panel {{
    background: {TAD_THEME['panel']};
    padding: 25px;
    border-radius: 15px;
    border: 1px solid {TAD_THEME['primary']};
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}}

.footer {{
    text-align: center;
    margin-top: 25px;
    padding: 15px;
    background: rgba(30, 30, 30, 0.7);
    border-radius: 12px;
    font-size: 0.9em;
}}

button {{ 
    transition: all 0.3s ease !important; 
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
button:hover {{ 
    transform: scale(1.03) !important; 
    box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3) !important;
}}

.textbox {{
    border-radius: 8px !important;
}}
"""


# ======================
# 🌍 TAD TRANSLATOR
# ======================
class TadTranslator:
    def __init__(self):
        self.models = self._load_models()

    def _load_models(self):
        """Professional model loader with visual flair"""
        model_map = {
            "en_to_sn": Path("turbo_models/english_to_shona"),
            "en_to_ve": Path("turbo_models/english_to_venda")
        }

        loaded_models = {}
        for name, path in model_map.items():
            try:
                if (path / "model.safetensors").exists():
                    pipe = pipeline(
                        "translation",
                        model=str(path),
                        device=0 if torch.cuda.is_available() else -1
                    )
                    loaded_models[name] = pipe
                    print(f"✨ {name.replace('_', ' ').upper()} model shining bright!")
                else:
                    print(f"🌑 Model not found at {path}")
            except Exception as e:
                print(f"🔥 Model loading fire: {str(e)}")
        return loaded_models

    def translate(self, text, direction):
        """Cultural translation with finesse"""
        if not text.strip():
            return "🌾 Please sow your words..."

        if direction not in self.models:
            return "⏳ Patience... model is still learning"

        try:
            result = self.models[direction](
                text,
                max_length=150,
                num_beams=4,
                temperature=0.7,
                no_repeat_ngram_size=2
            )
            translation = result[0]['translation_text']

            # Cultural tags
            lang_icon = "🌿" if direction == "en_to_sn" else "🏔️"
            return f"{lang_icon} {translation.capitalize()}"

        except Exception as e:
            return f"🌧️ Translation shower: {str(e)}"


# ======================
# 💫 GOLDEN INTERFACE
# ======================
def create_ui(translator):
    with gr.Blocks(
            theme=gr.themes.Default(
                primary_hue="amber",
                secondary_hue="emerald",
                neutral_hue="stone"
            ),
            css=CUSTOM_CSS
    ) as app:
        # 🌟 Cultural Header
        with gr.Column(elem_classes="header"):
            gr.Markdown("""
            <h1 style="margin-bottom: 0;">
            <img src="https://emojicdn.elk.sh/🌍" width="40" height="40" 
                 style="vertical-align: middle; filter: drop-shadow(0 0 4px #FFD700)">
            TAD TRANSLATOR
            <img src="https://emojicdn.elk.sh/🗣️" width="40" height="40" 
                 style="vertical-align: middle; filter: drop-shadow(0 0 4px #009739)">
            </h1>
            <p style="color: #AAAAAA; margin-top: 0;">
            Bridging languages with golden precision
            </p>
            """)

        # 🏡 Main Translation Hut
        with gr.Row():
            with gr.Column(scale=1, elem_classes="translation-panel"):
                gr.Markdown("### 🌱 Plant Your Words")
                input_text = gr.Textbox(
                    label="English Text",
                    placeholder="Type your message here...",
                    lines=4,
                    max_lines=6
                )

                direction = gr.Radio(
                    choices=[
                        ("English to Shona 🌿", "en_to_sn"),
                        ("English to Venda 🏔️", "en_to_ve")
                    ],
                    label="Journey Direction",
                    value="en_to_sn",
                    elem_classes="radio-buttons"
                )

                translate_btn = gr.Button(
                    "Weave Translation →",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1, elem_classes="translation-panel"):
                gr.Markdown("### 🏆 Harvested Translation")
                output = gr.Textbox(
                    label="Result",
                    interactive=False,
                    lines=4,
                    show_copy_button=True
                )

        # 📜 Cultural Examples
        gr.Markdown("### 🎋 Traditional Phrases")
        gr.Examples(
            examples=[
                ["Good morning, how did you sleep?", "en_to_sn"],
                ["Where can we find fresh water?", "en_to_ve"],
                ["Thank you for your wisdom", "en_to_sn"],
                ["The harvest will be good this year", "en_to_ve"]
            ],
            inputs=[input_text, direction],
            outputs=output,
            fn=translator.translate,
            cache_examples=True,
            label=""
        )

        # 🏮 Cultural Footer
        with gr.Column(elem_classes="footer"):
            gr.Markdown("""
            *From the land of balanced rocks and golden sunsets*  
            [Our Story] | [Language Roots]  
            © 2023 Tad Language Technologies
            """)

        # Connect the cultural bridge
        translate_btn.click(
            fn=translator.translate,
            inputs=[input_text, direction],
            outputs=output
        )

    return app


def find_available_port(start_port=7860, end_port=7900):
    """Find an available port in range"""
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
            return port
        except OSError:
            continue
    return None


# ======================
# 🌄 LAUNCH CEREMONY
# ======================
if __name__ == "__main__":
    print("""
     __  __           _   _           _   _       _   
    |  \/  |_ __ ___ | | | |_ __ ___ | | | | __ _(_)  
    | |\/| | '__/ _ \| |_| | '__/ _ \| |_| |/ _` | |  
    | |  | | | | (_) |  _  | | | (_) |  _  | (_| | |  
    |_|  |_|_|  \___/|_| |_|_|  \___/|_| |_|\__,_|_|  
    """)

    print("Kindling the Tad Translator fire...\n")

    # Initialize translator
    translator = TadTranslator()
    app = create_ui(translator)

    # Find available port
    port = find_available_port()
    if port is None:
        print("❌ Error: Could not find an available port (7860-7900)")
    else:
        print(f"🌐 Serving on http://127.0.0.1:{port}")
        print("Press Ctrl+C to stop the server\n")

        try:
            app.launch(
                server_name="127.0.0.1",  # Correct localhost address
                server_port=port,
                share=False,
                favicon_path="https://emojicdn.elk.sh/🌍"
            )
        except KeyboardInterrupt:
            print("\n🔥 Server stopped by user")
        except Exception as e:
            print(f"\n❌ Launch failed: {str(e)}")
            print("Try these solutions:")
            print("1. Close other applications using port 7860")
            print("2. Run as administrator if on Windows")
            print("3. Try app.launch(share=True) for public link")