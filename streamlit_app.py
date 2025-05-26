import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# Load saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('saved_mental_status_bert')
tokenizer = AutoTokenizer.from_pretrained('saved_mental_status_bert')
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

with st.sidebar:
    st.image(r'MINDCARE.jpg')
    st.title("Mental Diseases")
    st.subheader(
        "Detection of mental health diseases for early detection and diagnosis."
        )

# Custom function to clean input
stop_words = set(stopwords.words('english'))

def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Prediction function with probabilities
def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, probabilities

# Sidebar: Accuracy and image
#st.sidebar.image(image, use_column_width=True)
#st.sidebar.markdown("### 🧠 Mental Health Status Detection")
#st.sidebar.markdown("This application uses a fine-tuned BERT model to detect your mental state from the text you enter.")
#st.sidebar.markdown("**Model Accuracy:** 92.6%")  # Replace with actual value

# Main title
st.title("Mental Health Status Detection with BERT")

# Text input
input_text = st.text_input("Enter Your mental state here....")

# Detect button
if st.button("detect"):
    predicted_class, probabilities = detect_anxiety(input_text)
    st.write("### 🧾 Predicted Status :", predicted_class)

    # Balloons for positive result
    if predicted_class.lower() == "normal":
        st.balloons()

    # Filter: only show classes with >1% confidence
    min_threshold = 0.01
    labels = label_encoder.classes_
    filtered_probs = []
    filtered_labels = []

    for prob, label in zip(probabilities, labels):
        if prob >= min_threshold:
            filtered_probs.append(prob)
            filtered_labels.append(f"{label} ({prob * 100:.1f}%)")

    # If all are too small, just show top 5
    if not filtered_probs:
        top_indices = np.argsort(probabilities)[-5:][::-1]
        filtered_probs = [probabilities[i] for i in top_indices]
        filtered_labels = [f"{labels[i]} ({probabilities[i] * 100:.1f}%)" for i in top_indices]

    # Pie chart
    fig, ax = plt.subplots(figsize=(7, 6))  # Bigger chart size
    colors = plt.cm.Paired(np.linspace(0, 1, len(filtered_probs)))

    wedges, texts, autotexts = ax.pie(
        filtered_probs,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        textprops={'fontsize': 10}
    )

    # Draw circle in center to make it a donut chart
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(centre_circle)

    # Add legend outside
    ax.legend(wedges, filtered_labels, title="Mental States", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
