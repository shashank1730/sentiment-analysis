import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model & tokenizer
MODEL_DIR = "models/distilbert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

st.title("🧠 Sentiment Analyzer")
st.markdown("Enter any text below and get its **sentiment** (positive or negative) with confidence.")

# Text input
user_input = st.text_area("💬 Your text", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenize
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs).item()
            confidence = torch.max(probs).item()

        sentiment = "✅ Positive" if predicted_class == 1 else "❌ Negative"
        st.markdown(f"### Sentiment: {sentiment}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
