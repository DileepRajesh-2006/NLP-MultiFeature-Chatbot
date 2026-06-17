

import streamlit as st


def show():



    import os
    import re
    import json
    import pickle
    import random
    import numpy as np
    import nltk

    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import load_model

    # Language detection
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        LANGDETECT_AVAILABLE = True
    except ImportError:
        LANGDETECT_AVAILABLE = False

    # Translation
    try:
        from deep_translator import GoogleTranslator
        TRANSLATE_AVAILABLE = True
    except ImportError:
        TRANSLATE_AVAILABLE = False

    nltk.download("punkt",     quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("punkt_tab", quiet=True)

    lemmatizer = WordNetLemmatizer()



    SUPPORTED_LANGUAGES = {
        "en": {"name": "English",  "flag": "🇬🇧", "native": "English"},
        "te": {"name": "Telugu",   "flag": "🇮🇳", "native": "తెలుగు"},
        "hi": {"name": "Hindi",    "flag": "🇮🇳", "native": "हिंदी"},
        "es": {"name": "Spanish",  "flag": "🇪🇸", "native": "Español"},
        "fr": {"name": "French",   "flag": "🇫🇷", "native": "Français"},
    }



    CULTURAL_GREETINGS = {
        "en": "Hello! How can I help you today?",
        "te": "నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?",
        "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?",
        "es": "¡Hola! ¿Cómo puedo ayudarte hoy?",
        "fr": "Bonjour! Comment puis-je vous aider aujourd'hui?",
    }



    CULTURAL_SENTIMENT_PREFIXES = {
        "Negative": {
            "en": "I'm sorry to hear that. ",
            "te": "అది విన్నందుకు చింతిస్తున్నాను. ",
            "hi": "यह सुनकर दुख हुआ। ",
            "es": "Lo siento mucho. ",
            "fr": "Je suis désolé d'entendre cela. ",
        },
        "Positive": {
            "en": "Great to hear! ",
            "te": "చాలా సంతోషంగా ఉంది! ",
            "hi": "यह सुनकर खुशी हुई! ",
            "es": "¡Me alegra escuchar eso! ",
            "fr": "Ravi de l'entendre! ",
        },
        "Neutral": {
            "en": "",
            "te": "",
            "hi": "",
            "es": "",
            "fr": "",
        }
    }



    MEDICAL_KEYWORDS = [
        # English
        "fever", "cold", "cough", "headache", "pain",
        "disease", "doctor", "hospital", "medicine",
        "treatment", "symptoms", "diabetes", "cancer",
        # Telugu
        "జ్వరం", "దగ్గు", "నొప్పి", "వైద్యుడు",
        "వ్యాధి", "ఆసుపత్రి", "మందు",
        # Hindi
        "बुखार", "खांसी", "दर्द", "डॉक्टर",
        "बीमारी", "अस्पताल", "दवा",
        # Spanish
        "fiebre", "tos", "dolor", "médico",
        "enfermedad", "hospital", "medicina",
        # French
        "fièvre", "toux", "douleur", "médecin",
        "maladie", "hôpital", "médicament",
    ]



    COMMON_SYMPTOM_RESPONSES = {
        "fever": (
            "**Managing a Fever at Home:**\n\n"
            "1. Rest and stay hydrated.\n"
            "2. Take acetaminophen or ibuprofen to reduce fever.\n"
            "3. Wear lightweight clothing.\n"
            "4. Apply a cool, damp cloth to your forehead.\n"
            "5. Seek medical help if fever is above 103°F or lasts more than 3 days."
        ),
        "cold": (
            "**Managing a Common Cold:**\n\n"
            "1. Rest as much as possible.\n"
            "2. Drink plenty of fluids.\n"
            "3. Use over-the-counter medications for symptom relief.\n"
            "4. Gargle with salt water for sore throat relief.\n"
            "5. Most colds resolve within 7-10 days."
        ),
        "cough": (
            "**Managing a Cough:**\n\n"
            "1. Stay hydrated — drink warm fluids.\n"
            "2. Use cough drops or throat lozenges.\n"
            "3. Try a teaspoon of honey.\n"
            "4. Use a humidifier.\n"
            "5. See a doctor if cough persists more than 3 weeks."
        ),
        "headache": (
            "**Managing a Headache:**\n\n"
            "1. Take ibuprofen or acetaminophen.\n"
            "2. Rest in a quiet, dark room.\n"
            "3. Apply a cold compress.\n"
            "4. Stay hydrated.\n"
            "5. See a doctor if headaches are severe or frequent."
        ),
    }



    @st.cache_resource
    def load_all_models():

        words   = pickle.load(open("words.pkl",   "rb"))
        classes = pickle.load(open("classes.pkl", "rb"))
        model   = load_model("chatbot_model.keras")

        with open("intents.json", "r") as f:
            intents = json.load(f)

        sentiment_model  = pickle.load(
            open("sentiment_model.pkl",  "rb")
        )
        tfidf_vectorizer = pickle.load(
            open("tfidf_vectorizer.pkl", "rb")
        )

        return (
            words, classes, model, intents,
            sentiment_model, tfidf_vectorizer
        )



    def detect_language(text):

        if not LANGDETECT_AVAILABLE:
            return "en"

        try:
            lang = detect(text)
            if lang in SUPPORTED_LANGUAGES:
                return lang
            return "en"
        except Exception:
            return "en"



    def translate_text(text, src_lang, dest_lang):

        if src_lang == dest_lang:
            return text

        if not TRANSLATE_AVAILABLE:
            return text

        try:
            translator = GoogleTranslator(
                source=src_lang if src_lang != "auto" else "auto",
                target=dest_lang
            )
            return translator.translate(text)
        except Exception:
            return text



    def predict_sentiment(text, sentiment_model, tfidf_vec):

        NEGATIVE_WORDS = [
            "pain", "fever", "cold", "headache",
            "sick", "ill", "suffering", "hurt",
            "నొప్పి", "జ్వరం", "दर्द", "बुखार",
            "dolor", "fièvre"
        ]

        if any(w in text.lower() for w in NEGATIVE_WORDS):
            return "Negative", "😟", 90.0

        # Translate to English for sentiment analysis
        english_text = translate_text(text, "auto", "en")

        cleaned = english_text.lower()
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        vector = tfidf_vec.transform([cleaned])

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

        if confidence < 45:
            label = "Neutral"
            emoji = "😐"

        return label, emoji, confidence



    def bag_of_words(sentence, words):

        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [
            lemmatizer.lemmatize(w.lower())
            for w in sentence_words
        ]

        bag = [0] * len(words)

        for sw in sentence_words:
            for i, word in enumerate(words):
                if word == sw:
                    bag[i] = 1

        return np.array(bag)

    def get_gym_response(sentence, model, words, classes, intents):

        bow    = bag_of_words(sentence, words)
        result = model.predict(np.array([bow]), verbose=0)[0]

        results = sorted(
            [[i, r] for i, r in enumerate(result) if r > 0.25],
            key=lambda x: x[1], reverse=True
        )

        if not results or float(results[0][1]) < 0.55:
            return None

        tag = classes[results[0][0]]

        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

        return None

    def get_common_symptom_response(text_en):

        text_lower = text_en.lower()

        for symptom, response in COMMON_SYMPTOM_RESPONSES.items():
            if symptom in text_lower:
                return response

        return None

    def is_medical_question(text):

        text_lower = text.lower()

        return any(kw in text_lower for kw in MEDICAL_KEYWORDS)



    def get_multilingual_response(
        user_input,
        detected_lang,
        model, words, classes, intents,
        sentiment_model, tfidf_vectorizer
    ):

        if detected_lang != "en":
            english_input = translate_text(
                user_input, detected_lang, "en"
            )
        else:
            english_input = user_input

        sentiment, emoji, confidence = predict_sentiment(
            user_input, sentiment_model, tfidf_vectorizer
        )

        prefix_en = CULTURAL_SENTIMENT_PREFIXES.get(
            sentiment, {}
        ).get("en", "")

        english_response = ""

        symptom_response = get_common_symptom_response(
            english_input
        )

        if symptom_response:
            english_response = prefix_en + symptom_response

        else:

            gym_response = get_gym_response(
                english_input, model, words, classes, intents
            )

            if gym_response:
                english_response = prefix_en + gym_response
            else:
                english_response = (
                    prefix_en +
                    "I'm not sure about that. "
                    "Please ask about gym, fitness, or health topics."
                )

        if detected_lang != "en":

            translated_response = translate_text(
                english_response, "en", detected_lang
            )

            cultural_prefix = CULTURAL_SENTIMENT_PREFIXES.get(
                sentiment, {}
            ).get(detected_lang, "")

            if prefix_en and translated_response.startswith(
                translate_text(prefix_en, "en", detected_lang)
            ):
                final_response = translated_response
            else:
                final_response = (
                    cultural_prefix + translated_response
                    if cultural_prefix
                    else translated_response
                )

        else:

            final_response = english_response

        return final_response, sentiment, emoji, confidence



    st.title("🌐 Multilingual Chatbot")

    st.markdown(
        "Chat in **English, Telugu, Hindi, Spanish, or French**! "
        "I automatically detect your language and respond accordingly."
    )

    st.markdown("---")


    lang_cols = st.columns(5)

    for i, (code, info) in enumerate(
        SUPPORTED_LANGUAGES.items()
    ):
        with lang_cols[i]:
            st.markdown(
                f"<div style='"
                f"text-align:center;"
                f"padding:8px;"
                f"border-radius:8px;"
                f"background:#f0f2f6'>"
                f"{info['flag']}<br>"
                f"<small>{info['native']}</small>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("")


    with st.sidebar:

        st.markdown("---")
        st.header("🌐 Language Settings")

        manual_lang = st.selectbox(
            "Force language (optional):",
            ["Auto Detect"] + [
                f"{v['flag']} {v['name']}"
                for v in SUPPORTED_LANGUAGES.values()
            ]
        )

        st.markdown("---")
        st.markdown("### 💡 Try these:")
        st.markdown(
            "**English:** Hello! I have a headache\n\n"
            "**Telugu:** నమస్కారం! నాకు జ్వరం వచ్చింది\n\n"
            "**Hindi:** नमस्ते! मुझे बुखार है\n\n"
            "**Spanish:** Hola! Tengo fiebre\n\n"
            "**French:** Bonjour! J'ai de la fièvre"
        )

        if not LANGDETECT_AVAILABLE:
            st.warning(
                "⚠️ langdetect not installed!\n"
                "Run: pip install langdetect"
            )

        if not TRANSLATE_AVAILABLE:
            st.warning(
                "⚠️ deep-translator not installed!\n"
                "Run: pip install deep-translator"
            )


    for key, default in {
        "multi_messages":   [],
        "detected_lang":    "en",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


    col1, col2 = st.columns([1, 5])

    with col1:
        if st.button("🗑️ Clear"):
            st.session_state.multi_messages = []
            st.session_state.detected_lang  = "en"
            st.rerun()

    st.markdown("---")


    try:
        (
            words, classes, model, intents,
            sentiment_model, tfidf_vectorizer
        ) = load_all_models()

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return


    if not st.session_state.multi_messages:

        with st.chat_message("assistant"):

            st.markdown(
                "👋 **Hello / నమస్కారం / नमस्ते / Hola / Bonjour!**\n\n"
                "I'm your multilingual assistant! "
                "Just type in any supported language and "
                "I'll automatically detect and respond in the same language.\n\n"
                "**Supported languages:**\n"
                "🇬🇧 English · 🇮🇳 Telugu · 🇮🇳 Hindi · "
                "🇪🇸 Spanish · 🇫🇷 French"
            )


    for msg in st.session_state.multi_messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if "lang" in msg and msg["role"] == "user":

                lang_info = SUPPORTED_LANGUAGES.get(
                    msg["lang"], {}
                )

                st.markdown(
                    f"<small style='color:gray'>"
                    f"🌐 Detected: "
                    f"{lang_info.get('flag', '')} "
                    f"{lang_info.get('name', msg['lang'])}"
                    f"</small>",
                    unsafe_allow_html=True
                )

            if "sentiment" in msg and msg["role"] == "user":

                color_map = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral":  "blue"
                }

                color = color_map.get(
                    msg["sentiment"], "blue"
                )

                st.markdown(
                    f"<small style='color:{color}'>"
                    f"Sentiment: {msg['emoji']} "
                    f"{msg['sentiment']} — "
                    f"{msg['confidence']}% confidence"
                    f"</small>",
                    unsafe_allow_html=True
                )


    user_input = st.chat_input(
        "Type in any language... "
        "/ ఏ భాషలోనైనా టైప్ చేయండి... "
        "/ किसी भी भाषा में टाइप करें..."
    )

    if user_input:

        user_input = user_input.strip()

        if manual_lang == "Auto Detect":
            detected_lang = detect_language(user_input)
        else:
            lang_name = manual_lang.split(" ", 1)[1]
            detected_lang = next(
                (
                    code for code, info in
                    SUPPORTED_LANGUAGES.items()
                    if info["name"] == lang_name
                ),
                "en"
            )

        st.session_state.detected_lang = detected_lang

        (
            bot_response,
            sentiment,
            emoji,
            confidence
        ) = get_multilingual_response(
            user_input,
            detected_lang,
            model, words, classes, intents,
            sentiment_model, tfidf_vectorizer
        )

        st.session_state.multi_messages.append({
            "role":       "user",
            "content":    user_input,
            "lang":       detected_lang,
            "sentiment":  sentiment,
            "emoji":      emoji,
            "confidence": confidence
        })

        st.session_state.multi_messages.append({
            "role":    "assistant",
            "content": bot_response,
            "lang":    detected_lang
        })

        st.rerun()