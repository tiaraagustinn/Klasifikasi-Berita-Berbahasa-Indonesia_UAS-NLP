import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Klasifikasi Berita IndoBERT", layout="wide")

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./model_save/")
    model = AutoModelForSequenceClassification.from_pretrained("./model_save/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    label_mapping = {0: 'bisnis', 1: 'bola', 2: 'news', 3: 'otomotif', 4: 'tekno'}
    return tokenizer, model, device, label_mapping

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def predict_category(text, tokenizer, model, device, label_mapping):
    text = preprocess_text(text)
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    return {
        'category': label_mapping[prediction],
        'probabilities': {label_mapping[i]: float(p) for i, p in enumerate(probs)}
    }

# UI Header
st.markdown("""
# 📰 Klasifikasi Berita Bahasa Indonesia
Klasifikasikan artikel berita ke dalam 5 kategori: **Bola**, **News**, **Bisnis**, **Tekno**, dan **Otomotif**.
""")

# Load model
tokenizer, model, device, label_mapping = load_model()

# --- INPUT AREA ---
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📥 Masukkan teks berita di sini:", height=200, key="text_input")

    if st.button("🔍 Klasifikasikan", key="predict_button"):
        if not text_input.strip():
            st.error("Teks tidak boleh kosong.")
        else:
            with st.spinner("Memproses..."):
                prediction = predict_category(text_input, tokenizer, model, device, label_mapping)
                st.markdown(f"### ✅ Kategori: **{prediction['category'].upper()}**")

                probs_df = pd.DataFrame({
                    'Kategori': list(prediction['probabilities'].keys()),
                    'Probabilitas': list(prediction['probabilities'].values())
                }).sort_values('Probabilitas', ascending=False)

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(
                    probs_df['Kategori'],
                    probs_df['Probabilitas'],
                    color=['#1f77b4' if cat == prediction['category'] else '#d3d3d3' for cat in probs_df['Kategori']]
                )
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{height:.2%}', ha='center', va='bottom')

                ax.set_ylim(0, 1)
                ax.set_ylabel("Probabilitas")
                ax.set_title("Probabilitas per Kategori")
                st.pyplot(fig)

# --- SIDEBAR EXAMPLE ---
with col2:
    st.markdown("### 🎯 Contoh Artikel")
    example_articles = {
        "Bola": "Timnas Indonesia akan bertanding melawan Malaysia...",
        "News": "Presiden Republik Indonesia meresmikan jalan tol...",
        "Bisnis": "Nilai tukar rupiah menguat terhadap dolar AS...",
        "Tekno": "Apple meluncurkan iPhone 16 dengan fitur AI...",
        "Otomotif": "Hyundai meluncurkan IONIQ 7 di Indonesia..."
    }
    selected_example = st.selectbox("Pilih contoh artikel:", list(example_articles.keys()))

    if st.button("Gunakan Contoh"):
        example_text = example_articles[selected_example]
        st.text_area("Contoh Teks:", example_text, height=160, key="example_area")

        with st.spinner("Mengklasifikasikan..."):
            prediction = predict_category(example_text, tokenizer, model, device, label_mapping)
            st.markdown(f"### ✅ Kategori: **{prediction['category'].upper()}**")

            probs_df = pd.DataFrame({
                'Kategori': list(prediction['probabilities'].keys()),
                'Probabilitas': list(prediction['probabilities'].values())
            }).sort_values('Probabilitas', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(
                probs_df['Kategori'],
                probs_df['Probabilitas'],
                color=['#1f77b4' if cat == prediction['category'] else '#d3d3d3' for cat in probs_df['Kategori']]
            )
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{height:.2%}', ha='center', va='bottom')

            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilitas")
            ax.set_title("Probabilitas per Kategori")
            st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Aplikasi ini dibuat untuk keperluan tugas NLP (SINF6054).**
Model yang digunakan: **IndoBERT fine-tuned**
""")
