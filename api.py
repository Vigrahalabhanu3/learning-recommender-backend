import pandas as pd
import numpy as np
import torch
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)
CORS(app)

# Load models
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_courses     = pd.read_csv("processed_data/cleaned_courses.csv")
recommender    = SentenceTransformer("./finetuned_recommender_v2")
bert_tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_finetuned")
bert_model     = DistilBertForSequenceClassification.from_pretrained("./distilbert_finetuned").to(device)
index          = faiss.read_index("processed_data/faiss_index.bin")

df_courses["rich_text"] = (
    "Course: " + df_courses["course_title"] + ". " +
    "Offered by: " + df_courses["course_organization"] + ". " +
    "Level: " + df_courses["course_difficulty"] + ". " +
    "Rating: " + df_courses["course_rating"].astype(str) + " out of 5."
)

@app.route("/recommend", methods=["POST"])
def recommend():
    data    = request.json
    profile = data.get("profile", "")

    # Predict level with DistilBERT
    inputs = bert_tokenizer(
        profile,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    pred_label      = torch.argmax(outputs.logits, dim=1).item()
    confidence      = torch.softmax(outputs.logits, dim=1).max().item()
    levels          = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
    predicted_level = levels[pred_label]

    # Get recommendations with Sentence Transformer
    profile_embedding  = recommender.encode([profile])
    profile_normalized = profile_embedding / np.linalg.norm(profile_embedding)
    scores, indices    = index.search(profile_normalized.astype(np.float32), 15)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        course = df_courses.iloc[idx]
        if course["course_difficulty"] == predicted_level:
            results.append({
                "title":        course["course_title"],
                "organization": course["course_organization"],
                "difficulty":   course["course_difficulty"],
                "rating":       float(course["course_rating"]),
                "match":        f"{score:.2%}",
                "link":         f"https://www.coursera.org/search?query={course['course_title'].replace(' ', '+')}"
            })
        if len(results) == 5:
            break

    # Fallback if no results
    if not results:
        for idx, score in zip(indices[0][:5], scores[0][:5]):
            course = df_courses.iloc[idx]
            results.append({
                "title":        course["course_title"],
                "organization": course["course_organization"],
                "difficulty":   course["course_difficulty"],
                "rating":       float(course["course_rating"]),
                "match":        f"{score:.2%}",
                "link":         f"https://www.coursera.org/search?query={course['course_title'].replace(' ', '+')}"
            })

    return jsonify({
        "level":      predicted_level,
        "confidence": f"{confidence:.2%}",
        "courses":    results
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "message": "API is working!"})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
