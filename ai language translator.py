import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

def load_model_and_tokenizer(source_lang, target_lang):
    """
    Load MarianMT model and tokenizer for a specific source and target language.
    """
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        return None, None
    return model, tokenizer

def translate_text(text, source_lang, target_lang):
    """
    Translate text from source language to target language using the MarianMT model.
    """
    model, tokenizer = load_model_and_tokenizer(source_lang, target_lang)
    if model is None or tokenizer is None:
        return "Model not available for this language pair."

    # Tokenize the text
    tokenized_text = tokenizer([text], return_tensors="pt", padding=True)

    # Generate translation
    translated = model.generate(**tokenized_text)

    # Decode the translation
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_text[0]
language_options = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Arabic": "ar",
    "Turkish": "tr",
    "Korean": "ko",
    "Hindi": "hi"
}

# Gradio interface function
def translate_interface(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text  # No translation needed if source and target languages are the same
    return translate_text(text, language_options[source_lang], language_options[target_lang])

# Define the Gradio interface
interface = gr.Interface(
    fn=translate_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text"),
        gr.Dropdown(choices=list(language_options.keys()), value="English", label="Source Language"),
        gr.Dropdown(choices=list(language_options.keys()), value="French", label="Target Language")
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="AI-Powered Language Translator",
    description="Translate text from one language to another using Hugging Face's MarianMT model."
)

# Launch the Gradio interface
interface.launch()
