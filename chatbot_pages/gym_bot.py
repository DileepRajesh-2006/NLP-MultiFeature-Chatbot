

import streamlit as st


def show():

    import random
    import json
    import pickle
    import numpy as np
    import nltk
    import re
    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import load_model

    nltk.download('punkt',     quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('punkt_tab', quiet=True)

    lemmatizer = WordNetLemmatizer()



    @st.cache_resource
    def load_models():

        words   = pickle.load(open("words.pkl",   "rb"))
        classes = pickle.load(open("classes.pkl", "rb"))
        model   = load_model("chatbot_model.keras")

        sentiment_model  = pickle.load(open("sentiment_model.pkl",  "rb"))
        tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

        with open("intents.json", "r") as f:
            intents = json.load(f)

        return (
            words, classes, model,
            sentiment_model, tfidf_vectorizer,
            intents
        )



    def preprocess_text(text):

        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_up_sentence(sentence):

        sentence_words = nltk.word_tokenize(sentence)

        return [
            lemmatizer.lemmatize(w.lower())
            for w in sentence_words
        ]

    def bag_of_words(sentence, words):

        sentence_words = clean_up_sentence(sentence)

        bag = [0] * len(words)

        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1

        return np.array(bag)



    def predict_sentiment(text, sentiment_model, tfidf_vectorizer):

        cleaned = preprocess_text(text)
        vector  = tfidf_vectorizer.transform([cleaned])

        prediction    = sentiment_model.predict(vector)[0]
        probabilities = sentiment_model.predict_proba(vector)[0]
        confidence    = round(max(probabilities) * 100, 1)

        sentiment_map = {
            0: ("Negative", "😟"),
            1: ("Positive", "😊"),
            2: ("Neutral",  "😐")
        }

        label, emoji = sentiment_map.get(
            prediction, ("Neutral", "😐")
        )

        if confidence < 50.0:
            label = "Neutral"
            emoji = "😐"

        return label, emoji, confidence



    def predict_class(sentence, model, words, classes):

        bow = bag_of_words(sentence, words)

        result = model.predict(
            np.array([bow]), verbose=0
        )[0]

        results = [
            [i, r]
            for i, r in enumerate(result)
            if r > 0.25
        ]

        results.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "intent":      classes[r[0]],
                "probability": str(r[1])
            }
            for r in results
        ]

    def get_response(intents_list, intents):

        if not intents_list:
            return "Sorry, I didn't understand that."

        if float(intents_list[0]["probability"]) < 0.40:
            return "Can you explain that differently?"

        tag = intents_list[0]["intent"]

        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

        return "Sorry, something went wrong."



    def build_sentiment_response(sentiment, bot_response):

        if sentiment == "Negative":

            negative_count = st.session_state.get(
                "negative_count", 0
            ) + 1

            st.session_state.negative_count = negative_count

            if negative_count >= 3:
                prefix = (
                    "I can see you've been going through "
                    "a tough time. Please don't hesitate "
                    "to seek professional help if needed. "
                )
            else:
                prefix = "I'm sorry to hear that. "

            return prefix + bot_response

        elif sentiment == "Positive":

            st.session_state.negative_count = 0
            return "Great to hear! " + bot_response

        else:

            st.session_state.negative_count = 0
            return bot_response



    st.title("🏋️ GYM Bot")
    st.markdown(
        "Your personal fitness and health assistant. "
        "Ask me anything about gym, injuries, or wellness!"
    )

    # Top buttons
    col1, col2 = st.columns([1, 6])

    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.gym_messages  = []
            st.session_state.negative_count = 0
            st.rerun()

    with col2:
        if st.button("💾 Save Chat"):
            if not st.session_state.get("gym_messages"):
                st.warning("No chat history to save.")
            else:
                import datetime
                lines = [
                    f"Chat saved on {datetime.datetime.now()}\n"
                ]
                for msg in st.session_state.gym_messages:
                    role = "You" if msg["role"] == "user" else "Bot"
                    lines.append(f"{role}: {msg['content']}")
                    if "sentiment" in msg:
                        lines.append(
                            f"Sentiment: {msg['emoji']} "
                            f"{msg['sentiment']} "
                            f"({msg['confidence']}%)"
                        )
                    lines.append("")
                st.download_button(
                    label="📥 Download",
                    data="\n".join(lines),
                    file_name="chat_history.txt",
                    mime="text/plain"
                )

    st.markdown("---")

    if "gym_messages" not in st.session_state:
        st.session_state.gym_messages = []

    if "negative_count" not in st.session_state:
        st.session_state.negative_count = 0

    try:
        (
            words, classes, model,
            sentiment_model, tfidf_vectorizer,
            intents
        ) = load_models()

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Display chat history
    for msg in st.session_state.gym_messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if "sentiment" in msg:

                color_map = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral":  "blue"
                }

                color = color_map.get(msg["sentiment"], "blue")

                st.markdown(
                    f"<small style='color:{color}'>"
                    f"Sentiment: {msg['emoji']} "
                    f"{msg['sentiment']} — "
                    f"{msg['confidence']}% confidence"
                    f"</small>",
                    unsafe_allow_html=True
                )

    user_input = st.chat_input("Type your message here...")

    if user_input:

        user_input = user_input.strip()

        sentiment, emoji, confidence = predict_sentiment(
            user_input, sentiment_model, tfidf_vectorizer
        )

        st.session_state.gym_messages.append({
            "role":       "user",
            "content":    user_input,
            "sentiment":  sentiment,
            "emoji":      emoji,
            "confidence": confidence
        })

        if user_input.lower() in ["exit", "quit", "bye"]:
            bot_response = "Goodbye! Have a great day! 👋"
        else:
            intents_list = predict_class(
                user_input, model, words, classes
            )
            bot_response = get_response(intents_list, intents)
            bot_response = build_sentiment_response(
                sentiment, bot_response
            )

        st.session_state.gym_messages.append({
            "role":    "assistant",
            "content": bot_response
        })

        st.rerun()