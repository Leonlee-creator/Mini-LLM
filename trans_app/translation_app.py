import os
import torch
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import time
import json
import traceback



def create_theme():
    try:
        theme = gr.themes.Default(
            primary_hue="green",
            secondary_hue="orange",
            neutral_hue="stone"
        ).set(
            body_background_fill='#F8F5F0',
            button_primary_background_fill='#2C5E1A',
            button_secondary_background_fill='#5D3A1F',
            button_secondary_text_color='#FFFFFF',
            background_fill_secondary='#E8D5B5',
            border_color_accent='#D1C7B7'
        )
        return theme
    except Exception as e:
        print(f"Error creating theme: {str(e)}")
        return gr.themes.Default()


# Define theme and CSS at the top level
APP_THEME = create_theme()
CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    font-family: 'Segoe UI', Roboto, sans-serif;
}
"""

class AppConfig:
    HISTORY_FILE = "translation_history.json"
    MAX_HISTORY = 50
    MODEL_PATHS = {
        "en_to_sn": "turbo_models/english_to_shona",
        "sn_to_en": "turbo_models/shona_to_english",
        "en_to_ve": "turbo_models/english_to_venda",
    }
    TRANSLATION_PARAMS = {
        "max_length": 150,
        "num_beams": 5,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "temperature": 0.7
    }


class LanguageTranslator:
    def __init__(self):
        print("\n🔥 Initializing translator...")
        self.models = {}
        self.tokenizers = {}
        self.history = []

        try:
            self._verify_model_paths()
            self._load_models()
            self._load_history()
            print("✅ Translator initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize translator: {str(e)}")
            traceback.print_exc()
            raise

    def _verify_model_paths(self):
        print("\n🔍 Verifying model paths:")
        for name, path in AppConfig.MODEL_PATHS.items():
            model_path = Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path.absolute()}")
            print(f"✅ Verified {name} at {model_path}")

    def _load_models(self):
        print("\n🔧 Loading models:")
        for name, path in AppConfig.MODEL_PATHS.items():
            try:
                start = time.time()
                print(f"⏳ Loading {name.replace('_', ' ').title()}...")
                self.tokenizers[name] = AutoTokenizer.from_pretrained(path)
                self.models[name] = AutoModelForSeq2SeqLM.from_pretrained(path)

                if torch.cuda.is_available():
                    self.models[name] = self.models[name].cuda()
                    print(f"✅ Loaded {name} to GPU in {time.time() - start:.1f}s")
                else:
                    print(f"✅ Loaded {name} to CPU in {time.time() - start:.1f}s")

                # Test the model
                test_text = "Hello" if name.startswith("en_to") else "Mhoro"
                inputs = self.tokenizers[name](test_text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                outputs = self.models[name].generate(**inputs)
                print(
                    f"Test translation ({name}): {self.tokenizers[name].decode(outputs[0], skip_special_tokens=True)}")

            except Exception as e:
                print(f"❌ Failed to load {name}: {str(e)}")
                traceback.print_exc()
                raise

    def _load_history(self):
        try:
            if os.path.exists(AppConfig.HISTORY_FILE):
                with open(AppConfig.HISTORY_FILE, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"⚠️ Couldn't load history: {str(e)}")
            self.history = []



def create_app_interface():
    print("\n🖥️ Building interface...")
    try:
        translator = LanguageTranslator()

        def translate(text, direction):
            direction_map = {
                "English to Shona": "en_to_sn",
                "Shona to English": "sn_to_en",
                "English to Venda": "en_to_ve"
            }
            model_key = direction_map.get(direction)
            if not model_key or not text:
                return ""

            try:
                inputs = translator.tokenizers[model_key](
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )

                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')

                outputs = translator.models[model_key].generate(
                    **inputs,
                    max_length=150,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

                return translator.tokenizers[model_key].decode(
                    outputs[0],
                    skip_special_tokens=True
                )
            except Exception as e:
                print(f"Translation error: {str(e)}")
                traceback.print_exc()
                return f"Error: {str(e)}"

        with gr.Blocks(css=CUSTOM_CSS, theme=APP_THEME, title="Shona & Venda Translator") as app:
            gr.Markdown("## 🤖 Welcome to your Shona & Venda Translator")
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(label="Input Text", lines=5)
                    direction = gr.Dropdown(
                        choices=["English to Shona", "Shona to English", "English to Venda"],
                        label="Translation Direction",
                        value="English to Shona"
                    )
                    translate_btn = gr.Button("Translate", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(label="Translation", lines=5, interactive=False)

            translate_btn.click(
                fn=translate,
                inputs=[input_text, direction],
                outputs=output_text
            )

            gr.Examples(
                examples=[
                    ["Hello, how are you?", "English to Shona"],
                    ["Mhoro, makadii?", "Shona to English"],
                    ["Good morning", "English to Venda"]
                ],
                inputs=[input_text, direction],
                outputs=output_text,
                fn=translate,
                cache_examples=True
            )

        return app

    except Exception as e:
        print(f"❌ Interface build failed: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("""
    ███████╗██╗  ██╗ ██████╗ ███╗   ██╗ █████╗ 
    ██╔════╝██║  ██║██╔═══██╗████╗  ██║██╔══██╗
    ███████╗███████║██║   ██║██╔██╗ ██║███████║
    ╚════██║██╔══██║██║   ██║██║╚██╗██║██╔══██║
    ███████║██║  ██║╚██████╔╝██║ ╚████║██║  ██║
    ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝
    """)

    try:
        print("🚀 Starting application...")
        app = create_app_interface()
        if app is not None:
            print("\n🌍 Server starting at http://127.0.0.1:7860")
            print("Press Ctrl+C to stop the server")
            app.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=False,
                show_error=True
            )
        else:
            print("❌ Failed to create application interface")
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        traceback.print_exc()
        print("\nTroubleshooting steps:")
        print("1. Verify model directories exist in turbo_models/")
        print("2. Check each model directory contains required files")
        print("3. Run 'pip install -r requirements.txt'")
        print("4. Check GPU compatibility if using CUDA")