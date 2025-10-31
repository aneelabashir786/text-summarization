import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# =======================================
# Streamlit Page Configuration
# =======================================
st.set_page_config(page_title="Text Summarizer", page_icon="📰", layout="wide")

st.title("📰 Text Summarizer using Fine-tuned T5 Model")
st.write("Enter a long article or paragraph and get a concise summary!")

# =======================================
# Load Model and Tokenizer (with fallback)
# =======================================
@st.cache_resource
def load_model():
    model_name = "aneelaBashir22f3414/summarization"
    local_dir = "summarization_model"

    # Try loading from local cache first (for offline use)
    if os.path.exists(local_dir):
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
        return tokenizer, model

    try:
        # Try online download from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            force_download=True,
            local_files_only=False
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            force_download=True,
            local_files_only=False
        )

        # Save locally for future offline runs
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        return tokenizer, model

    except Exception as e:
        st.error("⚠️ Error loading model. Please check your internet connection or model path.")
        st.stop()

tokenizer, model = load_model()

# =======================================
# User Input
# =======================================
article = st.text_area("Enter Article:", height=250, placeholder="Paste your text here...")

# =======================================
# Generate Summary
# =======================================
if st.button("Generate Summary"):
    if article.strip():
        with st.spinner("⏳ Generating summary... please wait"):
            try:
                inputs = tokenizer(
                    article,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )

                # Move model to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                input_ids = inputs["input_ids"].to(device)

                summary_ids = model.generate(
                    input_ids,
                    max_length=150,
                    min_length=40,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )

                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                st.subheader("🧾 Summary:")
                st.success(summary)

            except Exception as e:
                st.error(f"⚠️ Error during summarization: {str(e)}")
    else:
        st.warning("Please enter some text before generating a summary!")

# =======================================
# Footer
# =======================================
st.markdown("---")
st.markdown("Developed with ❤️ using **Streamlit** & **Hugging Face Transformers**")
