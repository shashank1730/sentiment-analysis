# 📊 Sentiment Analysis with DistilBERT

This project builds a real-time **Sentiment Classification** system using a fine-tuned DistilBERT model on Amazon Reviews. It includes:

- A transformer-based model trained on 160K balanced reviews
- A baseline model using TF-IDF + Logistic Regression
- A **Streamlit app** for live predictions
- Docker support for easy deployment

---

## 🚀 Features

- **Model Accuracy**: Achieved F1 score of **0.943** on validation set
- **Interactive UI**: Enter text, get prediction and confidence score
- **Lightweight API**: Runs via Streamlit with Dockerized setup
- **Baseline Comparison**: TF-IDF + Logistic Regression for reference

---

## 🧠 Model Architecture

- Base: `distilbert-base-uncased`
- Fine-tuned for binary classification (positive/negative)
- Early stopping based on validation F1 score
- Trained using Hugging Face `Trainer`

---

## 📁 Project Structure

sentiment-analysis/
│
├── app/ # Streamlit frontend
│ └── app.py
├── data/ # Local dataset (not tracked in Git)
├── models/ # Trained model (mounted via volume)
├── notebooks/ # Preprocessing, training notebooks
├── requirements.txt
├── Dockerfile
└── README.md

---

## 🧪 Quickstart (Local with Streamlit)

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/app.py

# 1. Build the Docker image
docker build -t sentiment-app .

# 2. Run the container (mounts the local model folder)
docker run -p 8501:8501 -v %cd%/models:/app/models sentiment-app


Input: "This product is amazing. I’m so happy with it!"
→ Output: Positive (Confidence: 98%)

Input: "Terrible experience, total waste of money."
→ Output: Negative (Confidence: 96%)
```
