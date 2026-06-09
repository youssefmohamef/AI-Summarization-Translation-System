import os
import sys
from transformers import BartForConditionalGeneration, BartTokenizer, MarianMTModel, MarianTokenizer

# Disable Hugging Face hub symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configure Hugging Face model names for summarization and translation
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-ar"

def load_models():
    """Load the summarization and translation models along with their respective tokenizers."""
    print("⏳ Loading AI models (this may take a few moments on the first run)...")
    try:
        # Load the BART summarization model and tokenizer
        sum_tokenizer = BartTokenizer.from_pretrained(SUMMARIZATION_MODEL)
        sum_model = BartForConditionalGeneration.from_pretrained(SUMMARIZATION_MODEL)
        
        # Load the MarianMT translation model and tokenizer
        trans_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
        trans_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL)
        
        print("✅ All models loaded successfully.")
        return sum_model, sum_tokenizer, trans_model, trans_tokenizer
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        sys.exit(1)

def summarize_text(text, model, tokenizer):
    """Generate an abstractive summary of the English input text using BART."""
    print("⏳ Summarizing text...")
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_tokens = model.generate(
        inputs["input_ids"], 
        max_length=130, 
        min_length=30, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(summary_tokens[0], skip_special_tokens=True)

def translate_text(text, model, tokenizer):
    """Translate the summarized English text into Arabic using MarianMT (Neural Machine Translation)."""
    print("⏳ Translating summary to Arabic...")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_tokens = model.generate(input_ids)
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translation

def get_input_text(file_path):
    """Read input text from a local file, falling back to a default CORD-19 sample if the file is missing."""
    default_text = (
        "Dataset Description: In response to the COVID-19 pandemic, the White House and "
        "a coalition of leading research groups have prepared the COVID-19 Open Research "
        "Dataset (CORD-19). CORD-19 is a resource of over 47,000 scholarly articles, "
        "including over 36,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses."
    )
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                print(f"📖 Data successfully loaded from file: {file_path}")
                return file.read()
        except Exception as e:
            print(f"⚠️ Failed to read the file, falling back to default text. Reason: {e}")
            return default_text
    else:
        print(f"⚠️ Input text file not found at: {file_path}. Using default text instead.")
        return default_text

def main():
    # Load all models and tokenizers
    sum_model, sum_tokenizer, trans_model, trans_tokenizer = load_models()
    
    # Define the default input text file path
    file_path = "input_text.txt"
    
    # Read the input text
    raw_text = get_input_text(file_path)
    
    # 1. Perform abstractive summarization
    summary_english = summarize_text(raw_text, sum_model, sum_tokenizer)
    print("\n================== English Summary ==================")
    print(summary_english)
    
    # 2. Translate the generated summary to Arabic
    translation_arabic = translate_text(summary_english, trans_model, trans_tokenizer)
    print("\n================== Arabic Translation ==================")
    print(translation_arabic)

if __name__ == "__main__":
    main()