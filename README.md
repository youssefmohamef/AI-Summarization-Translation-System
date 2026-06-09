AI Summarization & Translation System 🚀

An advanced, dual-stage Natural Language Processing (NLP) pipeline designed to perform state-of-the-art Abstractive Summarization on complex English documents and generate highly precise Neural Machine Translations in Arabic.

This system leverages cutting-edge deep learning models (BART and MarianMT) from Hugging Face to deliver grammatically natural summaries and translations.

🌟 Key Features

Abstractive Summarization: Uses facebook/bart-large-cnn to synthesize dense, multi-paragraph documents (such as academic or medical literature like CORD-19) into original, context-aware summaries.

Neural Machine Translation (NMT): Integrates the Helsinki-NLP/opus-mt-en-ar MarianMT model to instantly convert English summaries into grammatically precise and natural Arabic text.

Robust Local Run-Guards: Features automated fallback logic that dynamically switches to pre-loaded sample datasets if the input text files are missing.

🛠️ Architecture Flow

Input Text: Load large text documents (e.g., CORD-19 medical dataset description).

Summarizer Stage (BART): Tokenizes and parses text through BartForConditionalGeneration to construct an optimized English summary.

Translator Stage (MarianMT): Takes the generated summary, processes it through neural sequence-to-sequence generation, and outputs the final translated Arabic text.

🚀 Getting Started

1. Clone the Repository

git clone [https://github.com/youssefmohamef/AI-Summarization-Translation-System.git](https://github.com/yourusername/AI-Summarization-Translation-System.git)
cd AI-Summarization-Translation-System


2. Install Dependencies

Make sure you have Python 3.8+ installed. Install the required deep learning and Hugging Face libraries:

pip install -r requirements.txt


3. Run the Pipeline

Create an input_text.txt file containing the document you want to summarize, or simply run the script to use the default built-in CORD-19 dataset sample:

python nlp_pipeline.py


📊 Sample Output

Input Text:

"In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19)..."

English Summary (BART):

"CORD-19 is a resource of over 47,000 scholarly articles, including over 36,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community."

Arabic Translation (MarianMT):

"وورد في CORD-19 أكثر من 47,000 مقال علمي، بما في ذلك أكثر من 36,000 مع نص كامل، عن COVID-19، وسارس-COV-2، وما يتصل بذلك من فيروسات الكورونا..."