import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
import pandas as pd

# 1. Load Model and Tokenizer (EN to FR)
@st.cache_resource
def load_nmt_model():
    # Using Helsinki-NLP's Transformer-based MarianMT
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_nmt_model()

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Medical NMT Evaluator", page_icon="💊")
st.title("🏥 Medical NMT & BLEU Evaluation")
st.markdown("### Neural Machine Translation System with Automated Metrics")

# Layout: Two columns for input
col1, col2 = st.columns(2)

with col1:
    source_input = st.text_area("Source Medical Text (EN):", 
                                "The patient was prescribed antibiotics for the pneumonia.")

with col2:
    reference_input = st.text_area("Reference Translation (FR):", 
                                 "Le patient a reçu des antibiotiques pour la pneumonie.\nLe patient s'est vu prescrire des antibiotiques pour la pneumonie.")
    st.caption("Tip: You can add multiple references (one per line) to improve evaluation accuracy.")

if st.button("Generate Translation & Calculate BLEU"):
    if source_input and reference_input:
        # Perform Translation
        with st.spinner('Translating using Transformer model...'):
            candidate = translate_text(source_input)
        
        # Prepare References (Split by newline for multiple candidates)
        refs_list = [line.strip() for line in reference_input.split('\n') if line.strip()]
        
        # Calculate BLEU using SacreBLEU
        # sacrebleu.corpus_bleu expects (list of candidates, list of lists of references)
        bleu = sacrebleu.corpus_bleu([candidate], [refs_list])
        
        # --- UI Display ---
        st.divider()
        st.subheader("Results")
        
        st.info(f"**NMT Output:** {candidate}")
        
        # Metric display for BLEU
        st.metric(label="Total BLEU Score", value=f"{bleu.score:.2f}")

        # N-Gram Precision Table
        st.write("### Modified N-Gram Precision")
        # bleu.precisions contains [p1, p2, p3, p4]
        n_gram_df = pd.DataFrame({
            "N-Gram Type": ["1-Gram (Unigram)", "2-Gram (Bigram)", "3-Gram (Trigram)", "4-Gram (4-gram)"],
            "Precision (%)": [f"{p:.2f}%" for p in bleu.precisions]
        })
        st.table(n_gram_df)

        # Detailed Breakdown
        with st.expander("View Evaluation Details (Brevity Penalty & Stats)"):
            st.write(f"- **Brevity Penalty (BP):** {bleu.bp:.4f}")
            st.write(f"- **Reference Length:** {bleu.ref_len}")
            st.write(f"- **System Output Length:** {bleu.sys_len}")
            st.write(f"- **Ratio:** {bleu.sys_len / bleu.ref_len:.4f}")
            if bleu.bp < 1.0:
                st.warning("The translation is shorter than the reference, triggering a Brevity Penalty.")
    else:
        st.error("Please provide both source text and at least one reference translation.")

# Instructions for the user/report
st.sidebar.header("Application Info")
st.sidebar.write("Architecture: **Transformer**")
st.sidebar.write("Model: **MarianMT**")
st.sidebar.write("Domain: **Healthcare/Medical**")