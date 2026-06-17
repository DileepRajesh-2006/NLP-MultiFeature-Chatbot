import os
import re
import pickle
import xml.etree.ElementTree as ET

import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ENTITY_KEYWORDS = {
    "Disease": [
        "cancer", "diabetes", "asthma", "arthritis", "alzheimer",
        "parkinson", "epilepsy", "stroke", "tuberculosis", "malaria",
        "hepatitis", "hiv", "aids", "lupus", "anemia", "leukemia",
        "lymphoma", "melanoma", "pneumonia", "bronchitis", "fever",
        "flu", "influenza", "covid", "cholera", "typhoid", "depression",
        "anxiety", "schizophrenia", "autism", "adhd", "hypertension",
        "obesity", "osteoporosis", "psoriasis", "eczema", "glaucoma",
        "cataract", "colitis", "crohn", "fibromyalgia", "gout",
        "migraine", "multiple sclerosis", "celiac", "sickle cell"
    ],

    "Symptom": [
        "pain", "fever", "cough", "fatigue", "nausea", "vomiting",
        "dizziness", "headache", "shortness of breath", "chest pain",
        "swelling", "rash", "itching", "bleeding", "bruising",
        "weight loss", "weight gain", "insomnia", "diarrhea",
        "constipation", "bloating", "cramps", "numbness", "tingling",
        "weakness", "confusion", "memory loss", "blurred vision",
        "hearing loss", "sore throat", "runny nose", "sneezing",
        "chills", "sweating", "loss of appetite", "back pain",
        "joint pain", "muscle pain", "abdominal pain", "seizures"
    ],

    "Treatment": [
        "surgery", "chemotherapy", "radiation", "medication",
        "therapy", "vaccine", "antibiotic", "insulin", "dialysis",
        "transplant", "physiotherapy", "immunotherapy", "hormone therapy",
        "blood transfusion", "biopsy", "endoscopy", "colonoscopy",
        "mri", "ct scan", "x-ray", "ultrasound", "ecg", "eeg",
        "rehabilitation", "counseling", "psychotherapy", "diet",
        "exercise", "supplement", "vitamins", "steroids", "antiviral",
        "antifungal", "painkiller", "ibuprofen", "acetaminophen"
    ]
}

def recognize_entities(text):
    text_lower = text.lower()
    found = {}

    for entity_type, keywords in ENTITY_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text_lower]

        if matches:
            found[entity_type] = matches

    return found


def parse_xml_file(filepath):
    qa_pairs = []

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for qa in root.findall(".//QAPair"):
            question_el = qa.find("Question")
            answer_el = qa.find("Answer")

            if question_el is None or answer_el is None:
                continue

            question = question_el.text
            answer = answer_el.text

            if not question or not answer:
                continue

            question = question.strip()
            answer = answer.strip()

            if len(answer) < 20:
                continue

            q_type = question_el.get("qtype", "general")

            focus = root.find(".//Focus")
            focus_text = (
                focus.text.strip()
                if focus is not None and focus.text
                else ""
            )

            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": q_type,
                "focus": focus_text
            })

    except Exception:
        pass

    return qa_pairs


@st.cache_resource
def load_medquad_data(_medquad_path):
    valid_folders = [
        "1_CancerGov_QA",
        "2_GARD_QA",
        "3_GHR_QA",
        "5_NIDDK_QA",
        "6_NINDS_QA",
        "7_SeniorHealth_QA",
        "8_NHLBI_QA_XML",
        "9_CDC_QA"
    ]

    all_qa = []

    for folder in valid_folders:
        folder_path = os.path.join(_medquad_path, folder)

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".xml"):
                continue

            filepath = os.path.join(folder_path, filename)
            qa_pairs = parse_xml_file(filepath)
            all_qa.extend(qa_pairs)

    return all_qa


@st.cache_resource
def build_retrieval_index(_qa_data):
    questions = [qa["question"] for qa in _qa_data]

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(questions)

    return vectorizer, tfidf_matrix


def retrieve_answer(
    user_query,
    qa_data,
    vectorizer,
    tfidf_matrix,
    top_k=1
):
    query_vec = vectorizer.transform([user_query])

    similarities = cosine_similarity(
        query_vec,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]
    top_score = similarities[top_indices[0]]

    if top_score < 0.1:
        return None, 0.0

    best_match = qa_data[top_indices[0]]

    return best_match, top_score


@st.cache_resource
def load_sentiment_model():
    sentiment_model = pickle.load(
        open("sentiment_model.pkl", "rb")
    )

    tfidf_vectorizer = pickle.load(
        open("tfidf_vectorizer.pkl", "rb")
    )

    return sentiment_model, tfidf_vectorizer


def predict_sentiment(text, sentiment_model, tfidf_vectorizer):
    text_clean = text.lower()
    text_clean = re.sub(r"[^\w\s]", "", text_clean).strip()

    vector = tfidf_vectorizer.transform([text_clean])

    prediction = sentiment_model.predict(vector)[0]
    probabilities = sentiment_model.predict_proba(vector)[0]
    confidence = round(max(probabilities) * 100, 1)

    sentiment_map = {
        0: ("Negative", "😟"),
        1: ("Positive", "😊"),
        2: ("Neutral", "😐")
    }

    label, emoji = sentiment_map.get(
        prediction,
        ("Neutral", "😐")
    )

    if confidence < 50.0:
        label = "Neutral"
        emoji = "😐"

    return label, emoji, confidence


def sentiment_prefix(sentiment):
    if sentiment == "Negative":
        return (
            "I understand you may be feeling anxious "
            "or concerned. Here is what I found:\n\n"
        )

    elif sentiment == "Positive":
        return (
            "Great that you're being proactive! "
            "Here is what I found:\n\n"
        )

    return ""


def show():
    st.title("🏥 Medical Q&A Chatbot")

    st.markdown(
        "Ask any medical question and I will find "
        "the most relevant answer from the "
        "**MedQuAD dataset** (47,457 NIH QA pairs)."
    )

    st.markdown("---")

    medquad_path = "MedQuAD-master"

    if not os.path.exists(medquad_path):
        st.error(
            f"MedQuAD dataset not found at "
            f"**{medquad_path}**.\n\n"
            "Please download from:\n"
            "https://github.com/abachaa/MedQuAD "
            "and extract into your project folder."
        )
        return

    with st.spinner("Loading MedQuAD dataset..."):
        qa_data = load_medquad_data(medquad_path)

    if not qa_data:
        st.error("No QA data loaded. Check your MedQuAD folder.")
        return

    with st.spinner("Building retrieval index..."):
        vectorizer, tfidf_matrix = build_retrieval_index(
            qa_data
        )

    try:
        sentiment_model, sent_vectorizer = load_sentiment_model()
        sentiment_available = True
    except Exception:
        sentiment_available = False

    st.success(
        f"✅ Loaded **{len(qa_data):,}** medical QA pairs"
    )

    if "med_messages" not in st.session_state:
        st.session_state.med_messages = []

    col1, col2 = st.columns([1, 6])

    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.med_messages = []
            st.rerun()

    st.markdown("---")

    for msg in st.session_state.med_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if "sentiment" in msg:
                color_map = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral": "blue"
                }

                color = color_map.get(
                    msg["sentiment"],
                    "blue"
                )

                st.markdown(
                    f"<small style='color:{color}'>"
                    f"Sentiment: {msg['emoji']} "
                    f"{msg['sentiment']} — "
                    f"{msg['confidence']}% confidence"
                    f"</small>",
                    unsafe_allow_html=True
                )

            if "entities" in msg and msg["entities"]:
                entity_str = " | ".join([
                    f"**{k}:** {', '.join(v)}"
                    for k, v in msg["entities"].items()
                ])

                st.markdown(
                    f"<small style='color:gray'>"
                    f"🔍 Entities: {entity_str}"
                    f"</small>",
                    unsafe_allow_html=True
                )

            if "score" in msg:
                st.markdown(
                    f"<small style='color:gray'>"
                    f"📊 Match confidence: "
                    f"{round(msg['score'] * 100, 1)}%"
                    f"</small>",
                    unsafe_allow_html=True
                )

    user_input = st.chat_input(
        "Ask a medical question..."
    )

    if user_input:
        user_input = user_input.strip()

        if sentiment_available:
            sentiment, emoji, confidence = predict_sentiment(
                user_input,
                sentiment_model,
                sent_vectorizer
            )
        else:
            sentiment, emoji, confidence = (
                "Neutral",
                "😐",
                0.0
            )

        entities = recognize_entities(user_input)

        st.session_state.med_messages.append({
            "role": "user",
            "content": user_input,
            "sentiment": sentiment,
            "emoji": emoji,
            "confidence": confidence,
            "entities": entities
        })

        best_match, score = retrieve_answer(
            user_input,
            qa_data,
            vectorizer,
            tfidf_matrix
        )

        if best_match is None:
            response = (
                "I'm sorry, I couldn't find a relevant "
                "answer in the MedQuAD dataset for your "
                "question. Please try rephrasing or ask "
                "a more specific medical question."
            )

            score = 0.0

        else:
            prefix = sentiment_prefix(sentiment)

            focus = best_match["focus"]
            q_type = best_match["type"]

            header = ""

            if focus:
                header += f"**Topic:** {focus}\n\n"

            if q_type and q_type != "general":
                header += (
                    f"**Question Type:** "
                    f"{q_type}\n\n"
                )

            response = (
                prefix
                + header
                + best_match["answer"]
            )

        st.session_state.med_messages.append({
            "role": "assistant",
            "content": response,
            "score": score
        })

        st.rerun()