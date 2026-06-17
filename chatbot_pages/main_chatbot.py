
# MAIN UNIFIED CHATBOT
# Task 1 : Sentiment Analysis
# Task 2 : Medical Q&A (MedQuAD)
# Task 3 : Dynamic Knowledge Base Expansion
# Task 4 : arXiv Research Assistant
# Task 5 : Multimodal — Image analysis via Gemini (📎 Attach Image)
# Task 6 : Multilingual Support


import streamlit as st


def show():

    import os
    import io
    import re
    import json
    import pickle
    import random
    import datetime
    import numpy as np
    import nltk
    import xml.etree.ElementTree as ET
    import requests
    from PIL import Image
    from collections import Counter

    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from tensorflow.keras.models import load_model

    from services.kb_search import load_vector_database, search_knowledge_base
    from services.kb_updater import build_vector_database

    try:
        from transformers import pipeline as hf_pipeline
        SUMMARIZER_AVAILABLE = True
    except ImportError:
        SUMMARIZER_AVAILABLE = False

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        LANGDETECT_AVAILABLE = True
    except ImportError:
        LANGDETECT_AVAILABLE = False

    try:
        from deep_translator import GoogleTranslator
        TRANSLATE_AVAILABLE = True
    except ImportError:
        TRANSLATE_AVAILABLE = False

    try:
        from google import genai
        from google.genai import types
        GEMINI_AVAILABLE = True
        NEW_SDK = True
    except ImportError:
        try:
            import google.generativeai as genai_old
            GEMINI_AVAILABLE = True
            NEW_SDK = False
        except ImportError:
            GEMINI_AVAILABLE = False
            NEW_SDK = False

    try:
        from groq import Groq
        GROQ_AVAILABLE = True
    except ImportError:
        GROQ_AVAILABLE = False

    nltk.download("punkt",     quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("punkt_tab", quiet=True)

    lemmatizer = WordNetLemmatizer()



    SUPPORTED_LANGUAGES = {
        "en": "English",
        "te": "Telugu",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
    }

    GEMINI_MODEL = "gemini-2.0-flash"

    MEDICAL_KEYWORDS = [
        "fever", "cold", "cough", "headache", "head ache",
        "pain", "fatigue", "vomiting", "nausea", "dizziness",
        "sore throat", "runny nose", "chills", "body pain",
        "weakness", "disease", "medicine", "doctor", "hospital",
        "treatment", "diagnosis", "symptoms", "infection",
        "virus", "bacteria", "diabetes", "cancer", "asthma",
        "arthritis", "covid", "flu", "influenza", "sick", "ill", "suffering",
        "జ్వరం", "దగ్గు", "నొప్పి", "వైద్యుడు", "వ్యాధి",
        "बुखार", "खांसी", "दर्द", "डॉक्टर", "बीमारी",
        "fiebre", "tos", "dolor", "médico", "enfermedad",
        "fièvre", "toux", "douleur", "médecin", "maladie",
    ]

    COMMON_ILLNESSES = [
        "common cold", "viral fever", "flu", "influenza",
        "migraine", "headache", "fever", "cough"
    ]

    ARXIV_KEYWORDS = [
        "paper", "papers", "research", "arxiv", "study", "studies",
        "publication", "journal", "preprint", "machine learning paper",
        "deep learning paper", "ai paper", "cs paper", "find paper",
        "search paper", "latest research", "recent research",
        "published paper", "scientific paper", "article on",
        "literature on", "survey on", "review paper",
    ]

    ARXIV_FOLLOWUP_KEYWORDS = [
        "summarize", "summary", "explain", "author",
        "pdf", "what is", "define", "tldr", "brief",
        "link", "download", "how does",
        # NEW — follow-up / analogy / continuation keywords
        "simpler", "analogy", "example", "elaborate",
        "tell me more", "what about", "and what",
        "key contributions", "methodology", "findings",
        "related work", "cite", "citation", "published",
        "when was", "who wrote", "contributions",
        "further", "more detail", "can you explain",
        "break it down", "in simple terms",
    ]

    COMMON_SYMPTOM_RESPONSES = {
        "fever": (
            "**Managing a Fever at Home:**\n\n"
            "1. Rest and stay hydrated — drink plenty of fluids.\n"
            "2. Take acetaminophen (Tylenol) or ibuprofen (Advil) to reduce fever.\n"
            "3. Wear lightweight clothing and use a light blanket.\n"
            "4. Apply a cool, damp cloth to your forehead.\n"
            "5. Seek medical help if fever is above 103°F (39.4°C), "
            "lasts more than 3 days, or is accompanied by severe symptoms."
        ),
        "cold": (
            "**Managing a Common Cold:**\n\n"
            "1. Rest as much as possible.\n"
            "2. Drink plenty of fluids — water, juice, or warm broth.\n"
            "3. Use over-the-counter medications for symptom relief.\n"
            "4. Gargle with salt water for sore throat relief.\n"
            "5. Use a humidifier to ease congestion.\n"
            "6. Most colds resolve within 7–10 days. See a doctor if symptoms worsen."
        ),
        "cough": (
            "**Managing a Cough:**\n\n"
            "1. Stay hydrated — drink warm fluids like honey and lemon tea.\n"
            "2. Use cough drops or throat lozenges.\n"
            "3. Try a teaspoon of honey (for adults and children over 1 year).\n"
            "4. Use a humidifier to add moisture to the air.\n"
            "5. Avoid smoking and secondhand smoke.\n"
            "6. See a doctor if cough persists more than 3 weeks."
        ),
        "headache": (
            "**Managing a Headache:**\n\n"
            "1. Take over-the-counter pain relievers like ibuprofen or acetaminophen.\n"
            "2. Rest in a quiet, dark room.\n"
            "3. Apply a cold or warm compress to your head or neck.\n"
            "4. Stay hydrated — dehydration is a common headache trigger.\n"
            "5. Avoid screen time and bright lights.\n"
            "6. See a doctor if headaches are severe, frequent, or accompanied by other symptoms."
        ),
        "nausea": (
            "**Managing Nausea:**\n\n"
            "1. Eat small, bland meals (crackers, toast, rice).\n"
            "2. Avoid strong smells, spicy, or fatty foods.\n"
            "3. Drink clear fluids slowly — ginger tea can help.\n"
            "4. Rest with your head elevated.\n"
            "5. See a doctor if nausea is severe or lasts more than 2 days."
        ),
        "vomiting": (
            "**Managing Vomiting:**\n\n"
            "1. Stop eating solid foods temporarily.\n"
            "2. Sip clear fluids slowly to stay hydrated.\n"
            "3. Gradually reintroduce bland foods (BRAT diet).\n"
            "4. Seek medical help if vomiting persists more than 24 hours."
        ),
        "dizziness": (
            "**Managing Dizziness:**\n\n"
            "1. Sit or lie down immediately to avoid falling.\n"
            "2. Move slowly when standing up.\n"
            "3. Stay hydrated and avoid caffeine and alcohol.\n"
            "4. Seek medical help if dizziness is severe or sudden."
        ),
    }

    ENTITY_KEYWORDS = {
        "Symptom": [
            "fever", "cold", "cough", "headache", "head ache",
            "pain", "fatigue", "vomiting", "nausea", "dizziness",
            "sore throat", "runny nose", "chills"
        ],
        "Disease": [
            "diabetes", "cancer", "asthma", "arthritis",
            "covid", "flu", "influenza"
        ]
    }

    IMAGE_QUESTION_KEYWORDS = [
        "describe", "what is", "what's in", "what are", "what do you see",
        "analyze", "analyse", "explain", "identify", "recognize", "recognise",
        "tell me about", "look at", "read", "text in", "words in",
        "color", "colour", "object", "person", "animal", "background",
        "how many", "where is", "what kind", "type of", "style of",
        "scene", "image", "picture", "photo", "this", "shown", "visible",
    ]



    def init_gemini():
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY or not GEMINI_AVAILABLE:
            return None
        try:
            if NEW_SDK:
                return genai.Client(api_key=GOOGLE_API_KEY)
            else:
                genai_old.configure(api_key=GOOGLE_API_KEY)
                return "old_sdk"
        except Exception:
            return None

    def gemini_image_analysis(prompt, image, gemini_client):
        if not gemini_client:
            return "⚠️ Gemini not available. Please add GOOGLE_API_KEY to your .env file."

        system_context = (
            "You are a helpful multimodal AI assistant. "
            "Analyze the provided image carefully and respond in detail."
        )
        full_prompt = f"{system_context}\n\nQuestion: {prompt}"

        try:
            if NEW_SDK and gemini_client != "old_sdk":
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=img_bytes,
                            )
                        ),
                        types.Part(text=full_prompt),
                    ],
                )
                return response.text
            else:
                m = genai_old.GenerativeModel(GEMINI_MODEL)
                return m.generate_content([full_prompt, image]).text

        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower() or "429" in err:
                return (
                    "⚠️ **Gemini quota reached.** Please wait and try again later.\n\n"
                    "*Image analysis requires the Gemini API.*"
                )
            return f"❌ Image analysis error: {e}"

    def is_image_question(text):
        return any(kw in text.lower() for kw in IMAGE_QUESTION_KEYWORDS)


    def detect_language(text):
        if len(text.split()) < 4:
            return "en"
        if not LANGDETECT_AVAILABLE:
            return "en"
        try:
            lang = detect(text)
            return lang if lang in SUPPORTED_LANGUAGES else "en"
        except Exception:
            return "en"

    def translate_text(text, src_lang, dest_lang):
        if src_lang == dest_lang or not TRANSLATE_AVAILABLE:
            return text
        try:
            return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
        except Exception:
            return text



    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @st.cache_resource
    def load_all_models():
        words   = pickle.load(open("words.pkl",   "rb"))
        classes = pickle.load(open("classes.pkl", "rb"))
        model   = load_model("chatbot_model.keras")
        with open("intents.json", "r") as f:
            intents = json.load(f)
        sentiment_model  = pickle.load(open("sentiment_model.pkl",  "rb"))
        tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        return (words, classes, model, intents, sentiment_model, tfidf_vectorizer)

    @st.cache_resource
    def load_medquad():
        medquad_path = "MedQuAD-master"
        cache_file   = "medquad_cache.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                all_qa = pickle.load(f)
        else:
            valid_folders = [
                "1_CancerGov_QA", "2_GARD_QA", "3_GHR_QA",
                "5_NIDDK_QA", "6_NINDS_QA", "7_SeniorHealth_QA",
                "8_NHLBI_QA_XML", "9_CDC_QA"
            ]
            all_qa = []
            for folder in valid_folders:
                folder_path = os.path.join(medquad_path, folder)
                if not os.path.exists(folder_path):
                    continue
                for filename in os.listdir(folder_path):
                    if not filename.endswith(".xml"):
                        continue
                    filepath = os.path.join(folder_path, filename)
                    try:
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for qa in root.findall(".//QAPair"):
                            q_el = qa.find("Question")
                            a_el = qa.find("Answer")
                            if q_el is None or a_el is None:
                                continue
                            question = q_el.text
                            answer   = a_el.text
                            if not question or not answer:
                                continue
                            if len(answer.strip()) < 40:
                                continue
                            focus = root.find(".//Focus")
                            focus_text = (
                                focus.text.strip()
                                if focus is not None and focus.text else ""
                            )
                            all_qa.append({
                                "question": question.strip(),
                                "answer":   answer.strip(),
                                "focus":    focus_text
                            })
                    except Exception:
                        continue
            with open(cache_file, "wb") as f:
                pickle.dump(all_qa, f)
        questions    = [qa["question"] for qa in all_qa]
        vectorizer   = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), stop_words="english"
        )
        tfidf_matrix = vectorizer.fit_transform(questions)
        return all_qa, vectorizer, tfidf_matrix


    def is_medical_question(text):

        text_lower = text.lower()
        if re.match(r'^what is\s+\w+\s*\??$', text_lower):
            # Only medical if the topic word is a medical keyword
            topic = re.sub(r'^what is\s+|\?$', '', text_lower).strip()
            medical_topics = [k for k in MEDICAL_KEYWORDS if k not in
                             ["what is", "how to treat", "what causes", "how is treated"]]
            return any(kw in topic for kw in medical_topics)
        return any(kw in text_lower for kw in MEDICAL_KEYWORDS)

    def get_common_symptom_response(text):
        text_lower = text.lower()
        for symptom, response in COMMON_SYMPTOM_RESPONSES.items():
            if symptom in text_lower:
                return response
        return None

    def recognize_entities(text):
        text_lower = text.lower()
        found = {}
        for entity_type, keywords in ENTITY_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                found[entity_type] = matches
        return found

    def predict_sentiment(text, sentiment_model, tfidf_vec):
        NEGATIVE_WORDS = [
            "pain", "fever", "cold", "headache", "sick", "ill",
            "suffering", "hurt", "నొప్పి", "జ్వరం", "दर्द", "बुखार",
            "dolor", "fièvre"
        ]
        if any(w in text.lower() for w in NEGATIVE_WORDS):
            return "Negative", "😟", 90.0
        cleaned = preprocess_text(text)
        vector  = tfidf_vec.transform([cleaned])
        prediction    = sentiment_model.predict(vector)[0]
        probabilities = sentiment_model.predict_proba(vector)[0]
        confidence    = round(max(probabilities) * 100, 1)
        sentiment_map = {
            0: ("Negative", "😟"),
            1: ("Positive", "😊"),
            2: ("Neutral",  "😐")
        }
        label, emoji = sentiment_map.get(prediction, ("Neutral", "😐"))
        if confidence < 45:
            label = "Neutral"
            emoji = "😐"
        return label, emoji, confidence

    def sentiment_prefix(sentiment, lang="en"):
        prefixes = {
            "Negative": {
                "en": "I'm sorry to hear that.\n\n",
                "te": "అది విన్నందుకు చింతిస్తున్నాను.\n\n",
                "hi": "यह सुनकर दुख हुआ।\n\n",
                "es": "Lo siento mucho.\n\n",
                "fr": "Je suis désolé d'entendre cela.\n\n",
            },
            "Positive": {
                "en": "Great to hear! \n\n",
                "te": "చాలా సంతోషంగా ఉంది! \n\n",
                "hi": "यह सुनकर खुशी हुई! \n\n",
                "es": "¡Me alegra escuchar eso! \n\n",
                "fr": "Ravi de l'entendre! \n\n",
            },
            "Neutral": {
                "en": "", "te": "", "hi": "", "es": "", "fr": ""
            }
        }
        negative_count = st.session_state.get("negative_count", 0)
        if sentiment == "Negative":
            negative_count += 1
            st.session_state.negative_count = negative_count
            if negative_count >= 3:
                msg = (
                    "I can see you've been going through a tough time. "
                    "Please seek professional help if needed.\n\n"
                )
                return translate_text(msg, "en", lang)
            return prefixes["Negative"].get(lang, prefixes["Negative"]["en"])
        elif sentiment == "Positive":
            st.session_state.negative_count = 0
            return prefixes["Positive"].get(lang, prefixes["Positive"]["en"])
        else:
            st.session_state.negative_count = 0
            return ""

    def bag_of_words(sentence, words):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
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

    def get_medical_response(query, qa_data, vectorizer, tfidf_matrix):
        query     = preprocess_text(query)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        simple_fever = query.strip() in [
            "fever", "normal fever", "common fever",
            "im suffering from fever", "i have fever", "i have a fever"
        ]
        for i, qa in enumerate(qa_data):
            focus    = qa["focus"].lower()
            question = qa["question"].lower()
            if "syndrome"    in focus: similarities[i] *= 0.2
            if "urticaria"   in focus: similarities[i] *= 0.2
            if "dengue"      in focus: similarities[i] *= 0.2
            if "q fever"     in focus: similarities[i] *= 0.1
            if "hemorrhagic" in focus: similarities[i] *= 0.2
            if simple_fever and "fever" in focus and focus != "fever":
                similarities[i] *= 0.1
            for illness in COMMON_ILLNESSES:
                if illness in focus:
                    similarities[i] *= 1.4
            if simple_fever and "treat" in question:
                similarities[i] *= 1.5
        top_idx   = similarities.argmax()
        top_score = similarities[top_idx]
        if top_score < 0.22:
            return None, 0.0
        return qa_data[top_idx], top_score

    def build_chat_export():
        lines = [f"Chat saved on {datetime.datetime.now()}\n"]
        for msg in st.session_state.main_messages:
            role = "You" if msg["role"] == "user" else "Bot"
            lines.append(f"{role}: {msg['content']}")
            if "sentiment" in msg:
                lines.append(
                    f"Sentiment: {msg['emoji']} {msg['sentiment']} ({msg['confidence']}%)"
                )
            if "lang" in msg:
                lines.append(
                    f"Language: {SUPPORTED_LANGUAGES.get(msg['lang'], msg['lang'])}"
                )
            lines.append("")
        return "\n".join(lines)



    def is_arxiv_question(text):
        return any(kw in text.lower() for kw in ARXIV_KEYWORDS)

    def search_arxiv(query, category="cs", max_results=5):
        search_query = f"all:{query}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            resp = requests.get(
                "https://export.arxiv.org/api/query",
                params=params, timeout=20
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return [], "CONNECTION_ERROR"
        except requests.exceptions.Timeout:
            return [], "TIMEOUT"
        except Exception as e:
            return [], str(e)

        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            def t(tag):
                el = entry.find(f"atom:{tag}", ns)
                return el.text.strip() if el is not None and el.text else ""
            raw_id  = t("id")
            authors = [
                a.find("atom:name", ns).text.strip()
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            categories = [
                c.get("term", "") for c in entry.findall("atom:category", ns)
            ]
            papers.append({
                "title":      t("title").replace("\n", " "),
                "abstract":   t("summary").replace("\n", " "),
                "authors":    authors,
                "published":  t("published")[:10],
                "url":        raw_id,
                "pdf_url":    raw_id.replace("/abs/", "/pdf/") + ".pdf",
                "categories": categories,
            })
        return papers, None

    def extract_search_topic(text):
        text_lower = text.lower()
        for phrase in [
            "find papers on", "find paper on", "search papers on",
            "search paper on", "research papers on", "papers on",
            "paper on", "latest research on", "recent research on",
            "find research on", "show papers on", "show paper on",
            "arxiv search", "arxiv", "find me papers about",
            "papers about", "research on", "survey on", "article on",
        ]:
            if phrase in text_lower:
                text_lower = text_lower.replace(phrase, "").strip()
        return text_lower.strip() or text.strip()

    def simple_extractive_summary(text, num_sentences=2):
        stopwords = {
            "the","a","an","in","of","on","and","or","for","to","with","is","are",
            "was","were","that","this","it","as","by","from","be","at","we","our",
            "have","has","which","their","these","can","also","not","but","more",
        }
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= num_sentences:
            return text
        words_all = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        freq      = Counter(w for w in words_all if w not in stopwords)
        keywords  = set(w for w, _ in freq.most_common(15))
        scores    = [
            len(set(re.findall(r'\b[a-zA-Z]{4,}\b', s.lower())) & keywords)
            for s in sentences
        ]
        top = sorted(
            sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        )
        return " ".join(sentences[i] for i in top)

    @st.cache_resource(show_spinner=False)
    def load_summarizer():
        if not SUMMARIZER_AVAILABLE:
            return None
        try:
            return hf_pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1
            )
        except Exception:
            return None

    def summarize_abstract(abstract):
        summarizer = load_summarizer()
        if summarizer:
            try:
                result = summarizer(
                    abstract, max_length=100, min_length=30, do_sample=False
                )
                return result[0]["summary_text"]
            except Exception:
                pass
        return simple_extractive_summary(abstract, num_sentences=2)

    # ── FIX 3: explain_concept with Groq fallback ─────────────────
    def explain_concept(concept):
        # Try Groq first (fast, free, intelligent)
        if GROQ_AVAILABLE:
            try:
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{
                        "role": "system",
                        "content": (
                            "You are an expert CS research assistant. "
                            "Give clear, accurate explanations for graduate students."
                        )
                    }, {
                        "role": "user",
                        "content": (
                            f"Explain '{concept}' in 3 sentences. "
                            "Be specific and technical but clear."
                        )
                    }],
                    max_tokens=200,
                    temperature=0.3,
                )
                answer = response.choices[0].message.content.strip()
                if answer:
                    return answer
            except Exception:
                pass

        try:
            payload = {
                "model":  "mistral",
                "prompt": (
                    f"You are an expert CS research assistant. "
                    f"Explain '{concept}' in 3 sentences for a graduate student."
                ),
                "stream": False,
            }
            resp = requests.post(
                "http://localhost:11434/api/generate", json=payload, timeout=15
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception:
            pass

        builtins = {
            "transformer": (
                "A Transformer is a deep learning architecture based on self-attention, "
                "introduced in 'Attention Is All You Need' (2017). It processes sequences "
                "in parallel, making it highly efficient for NLP tasks. "
                "Transformers power BERT, GPT, and T5."
            ),
            "attention": (
                "Attention mechanisms let neural networks focus on the most relevant parts "
                "of the input when making predictions. Self-attention computes relationships "
                "between all positions in a sequence simultaneously. "
                "It captures long-range dependencies without recurrence."
            ),
            "bert": (
                "BERT (Bidirectional Encoder Representations from Transformers) reads text "
                "bidirectionally using masked language modeling pre-training. "
                "It can be fine-tuned for downstream NLP tasks with minimal labeled data. "
                "BERT set new state-of-the-art results on 11 NLP benchmarks when released."
            ),
            "gpt": (
                "GPT is an autoregressive language model that predicts the next token "
                "given all previous tokens. It generates coherent text and scales "
                "remarkably well with data and compute. GPT-4 powers ChatGPT."
            ),
            "diffusion": (
                "Diffusion models learn to reverse a gradual noising process applied to data. "
                "Generation starts from pure Gaussian noise and iteratively denoises it "
                "into a clean data sample. They achieve state-of-the-art image synthesis results."
            ),
            "embedding": (
                "Embeddings map discrete objects (words, users, items) into a continuous "
                "vector space where semantic similarity corresponds to geometric proximity. "
                "They are learned during training and capture rich semantic relationships."
            ),
            "neural network": (
                "Neural networks are computational models organized in layers of interconnected "
                "nodes that learn abstract representations through backpropagation. "
                "Each layer transforms its input into increasingly abstract representations."
            ),
            "reinforcement learning": (
                "Reinforcement learning trains agents to maximize cumulative reward "
                "through trial and error interactions with an environment. "
                "The agent learns a policy mapping states to actions via exploration and exploitation."
            ),
            "quantum entanglement": (
                "Quantum entanglement is a phenomenon where two particles become correlated "
                "such that measuring one instantly determines the state of the other, "
                "regardless of the distance between them. "
                "It is a key resource in quantum computing and quantum cryptography."
            ),
            "crispr": (
                "CRISPR-Cas9 is a molecular tool that can precisely edit DNA sequences "
                "in living organisms. A guide RNA directs the Cas9 protein to a specific "
                "genomic location where it makes a cut, enabling gene addition, removal, or correction."
            ),
        }
        for key, explanation in builtins.items():
            if key in concept.lower():
                return explanation

        return (
            f"**{concept}** is a research concept. "
            f"Search arXiv for foundational papers on this topic using: "
            f"*find papers on {concept}*"
        )

    def get_arxiv_response(user_input):
        text_lower   = user_input.lower()
        active_paper = st.session_state.get("arxiv_active_paper", None)

        if any(w in text_lower for w in ["summarize", "summary", "tldr", "brief"]):
            if active_paper:
                summary = summarize_abstract(active_paper["abstract"])
                return (
                    f"📄 **Summary of '{active_paper['title'][:60]}...'**\n\n"
                    f"{summary}\n\n"
                    f"*{active_paper['published']} · "
                    f"{', '.join(active_paper['authors'][:2])}*"
                ), "arXiv · NLP Summary"
            return "No active paper to summarize. Try: *find papers on transformers*", "arXiv"

        if any(w in text_lower for w in ["explain", "what is", "what are", "define", "how does"]):
            concept = re.sub(
                r"(explain|what is|what are|define|definition of|how does)",
                "", text_lower
            ).strip().strip("?").strip()
            if concept:
                explanation = explain_concept(concept)
                return (
                    f"💡 **{concept.title()}**\n\n{explanation}",
                    "arXiv · Concept Explanation"
                )

        if "author" in text_lower and active_paper:
            return "👥 **Authors:** " + ", ".join(active_paper["authors"]), "arXiv"

        if any(w in text_lower for w in ["pdf", "link", "download"]) and active_paper:
            return (
                f"📎 **PDF:** [{active_paper['title'][:55]}...]({active_paper['pdf_url']})",
                "arXiv"
            )

        if active_paper and any(w in text_lower for w in [
            "simpler", "analogy", "example", "elaborate", "tell me more",
            "what about", "and what", "key contributions", "methodology",
            "findings", "related work", "cite", "citation",
            "further", "more detail", "can you explain",
            "break it down", "in simple terms"
        ]):
            concept = active_paper["title"]
            explanation = explain_concept(concept)
            return (
                f"💡 **Follow-up on: {active_paper['title'][:60]}**\n\n"
                f"{explanation}\n\n"
                f"*Ask me to `summarize`, get `authors`, or `pdf link`*"
            ), "arXiv · Follow-up"

        topic = extract_search_topic(user_input)
        with st.spinner(f"🔍 Searching arXiv for '{topic}'..."):
            papers, err = search_arxiv(topic, category="cs", max_results=5)

        if err or not papers:
            if err == "CONNECTION_ERROR":
                msg = f"❌ Cannot connect to arXiv for **{topic}**."
            elif err == "TIMEOUT":
                msg = f"⏱ arXiv timed out for **{topic}**. Try again."
            elif not papers:
                msg = f"🔍 No papers found for **{topic}**."
            else:
                msg = f"❌ arXiv error: `{err}`"
            return msg, "arXiv API Error"

        st.session_state.arxiv_active_paper = papers[0]
        st.session_state.arxiv_last_results = papers

        lines = [f"📚 Found **{len(papers)} papers** on **{topic}**:\n"]
        for i, p in enumerate(papers, 1):
            summary = simple_extractive_summary(p["abstract"], num_sentences=1)
            lines.append(
                f"**{i}. {p['title']}**\n"
                f"*{p['published']} · "
                f"{', '.join(p['authors'][:2])}{'...' if len(p['authors'])>2 else ''}*\n"
                f"{summary}\n"
                f"[📎 PDF]({p['pdf_url']})\n"
            )
        lines.append(
            "\n💡 **Ask me to:** `summarize` · `explain [concept]` · "
            "`authors` · `pdf link`\n"
            "*(Paper 1 is now active for follow-up questions)*"
        )
        return "\n".join(lines), f"arXiv API · {len(papers)} papers"



    st.title("🤖 AI Chatbot")
    st.markdown(
        "Ask me anything about 🏋️ **Gym & Fitness**, "
        "🩺 **Medical Questions**, "
        "📚 **Uploaded Documents**, "
        "🔬 **CS Research Papers**, "
        "🖼️ **Images (Gemini Vision)**, or "
        "🌐 **any language**!"
    )

    with st.sidebar:
        st.markdown("---")
        st.header("📚 Knowledge Base")
        uploaded_file = st.file_uploader(
            "Upload document (TXT, PDF, DOCX)",
            type=["txt", "pdf", "docx"]
        )
        if uploaded_file is not None:
            os.makedirs("knowledge_base/documents", exist_ok=True)
            save_path = os.path.join(
                "knowledge_base/documents", uploaded_file.name
            )
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} uploaded!")

        if st.button("🔄 Update Knowledge Base"):
            with st.spinner("Updating..."):
                try:
                    total = build_vector_database()
                    st.success(f"Updated! {total} chunks indexed.")
                except Exception as e:
                    st.error(f"Update failed: {e}")

        active = st.session_state.get("arxiv_active_paper", None)
        if active:
            st.markdown("---")
            st.markdown("### 🔬 Active Research Paper")
            st.caption(active["title"][:80] + "…")
            st.caption(f"📅 {active['published']}")
            if st.button("❌ Clear Active Paper"):
                st.session_state.arxiv_active_paper = None
                st.session_state.arxiv_last_results = []
                st.rerun()

        st.markdown("---")
        st.markdown("### 🌐 Language")
        detected  = st.session_state.get("detected_lang", "en")
        lang_name = SUPPORTED_LANGUAGES.get(detected, "English")
        st.info(f"Detected: **{lang_name}**")

        if st.session_state.get("attached_image") is not None:
            st.markdown("---")
            st.markdown("### 🖼️ Attached Image")
            st.image(
                st.session_state.attached_image,
                caption=st.session_state.get("attached_image_name", "Image"),
                use_container_width=True
            )
            st.caption("🔵 Gemini Vision active")
            if st.button("🗑️ Remove Image", use_container_width=True):
                st.session_state.attached_image      = None
                st.session_state.attached_image_name = None
                st.rerun()

    for key, default in {
        "main_messages":       [],
        "negative_count":      0,
        "show_download":       False,
        "arxiv_active_paper":  None,
        "arxiv_last_results":  [],
        "detected_lang":       "en",
        "attached_image":      None,
        "attached_image_name": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear"):
            st.session_state.main_messages       = []
            st.session_state.negative_count      = 0
            st.session_state.arxiv_active_paper  = None
            st.session_state.arxiv_last_results  = []
            st.session_state.detected_lang       = "en"
            st.session_state.attached_image      = None
            st.session_state.attached_image_name = None
            st.rerun()
    with col2:
        if st.button("💾 Save Chat"):
            st.session_state.show_download = True

    if st.session_state.show_download and st.session_state.main_messages:
        st.download_button(
            label="📥 Download Chat",
            data=build_chat_export(),
            file_name="chat_history.txt",
            mime="text/plain"
        )

    st.markdown("---")

    try:
        (words, classes, model, intents,
         sentiment_model, tfidf_vectorizer) = load_all_models()
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return

    medquad_available = os.path.exists("MedQuAD-master")
    if medquad_available:
        with st.spinner("Loading medical knowledge base..."):
            qa_data, med_vectorizer, med_tfidf = load_medquad()
    else:
        qa_data = med_vectorizer = med_tfidf = None

    try:
        vector_db = load_vector_database()
    except Exception:
        vector_db = None

    gemini_client = init_gemini()

    for msg in st.session_state.main_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("has_image") and msg["role"] == "user":
                st.image(
                    msg["image_data"],
                    caption=f"📎 {msg.get('image_name', 'Image')}",
                    width=200
                )

            if "sentiment" in msg:
                color_map = {
                    "Positive": "green", "Negative": "red", "Neutral": "blue"
                }
                color = color_map.get(msg["sentiment"], "blue")
                st.markdown(
                    f"<small style='color:{color}'>Sentiment: {msg['emoji']} "
                    f"{msg['sentiment']} — {msg['confidence']}% confidence</small>",
                    unsafe_allow_html=True
                )
            if msg.get("entities"):
                entity_text = " | ".join(
                    [f"**{k}:** {', '.join(v)}"
                     for k, v in msg["entities"].items()]
                )
                st.markdown(
                    f"<small style='color:gray'>🔍 {entity_text}</small>",
                    unsafe_allow_html=True
                )
            if msg.get("lang") and msg["role"] == "user":
                lang_name = SUPPORTED_LANGUAGES.get(msg["lang"], msg["lang"])
                if msg["lang"] != "en":
                    st.markdown(
                        f"<small style='color:gray'>🌐 {lang_name}</small>",
                        unsafe_allow_html=True
                    )
            if msg.get("source"):
                st.markdown(
                    f"<small style='color:gray'>📌 Source: {msg['source']}</small>",
                    unsafe_allow_html=True
                )

    with st.expander("📎 Attach Image", expanded=False):
        col_up, col_status = st.columns([3, 2])
        with col_up:
            img_upload = st.file_uploader(
                "Upload image for Gemini Vision analysis",
                type=["jpg", "jpeg", "png", "webp"],
                key="main_image_uploader",
                label_visibility="collapsed"
            )
            if img_upload:
                pil_img = Image.open(img_upload).convert("RGB")
                st.session_state.attached_image      = pil_img
                st.session_state.attached_image_name = img_upload.name
                st.success(f"✅ {img_upload.name} attached!")
        with col_status:
            if st.session_state.attached_image is not None:
                st.image(
                    st.session_state.attached_image,
                    caption=st.session_state.attached_image_name,
                    use_container_width=True
                )
                if st.button("❌ Remove", key="remove_img_expander"):
                    st.session_state.attached_image      = None
                    st.session_state.attached_image_name = None
                    st.rerun()
            else:
                st.info("No image attached.\nGemini will analyze when you attach one.")

    placeholder = (
        "Ask about the image, or type anything..."
        if st.session_state.attached_image is not None
        else "Ask me anything..."
    )
    user_input = st.chat_input(placeholder)

    if user_input:

        user_input = user_input.strip()

        current_image      = st.session_state.attached_image
        current_image_name = st.session_state.attached_image_name

        # FIX 1: improved language detection (min 4 words)
        detected_lang = detect_language(user_input)
        st.session_state.detected_lang = detected_lang

        english_input = (
            translate_text(user_input, detected_lang, "en")
            if detected_lang != "en" else user_input
        )

        sentiment, emoji, confidence = predict_sentiment(
            english_input, sentiment_model, tfidf_vectorizer
        )
        entities = recognize_entities(english_input)

        user_msg = {
            "role":       "user",
            "content":    user_input,
            "sentiment":  sentiment,
            "emoji":      emoji,
            "confidence": confidence,
            "entities":   entities,
            "lang":       detected_lang,
            "has_image":  current_image is not None,
            "image_data": current_image,
            "image_name": current_image_name,
        }
        st.session_state.main_messages.append(user_msg)

        prefix       = sentiment_prefix(sentiment, detected_lang)
        source       = ""
        bot_response = ""

        # ── Task 5: Image — highest priority ─────────────────────
        if current_image is not None and is_image_question(english_input):
            with st.spinner("🔵 Gemini Vision analyzing image..."):
                analysis = gemini_image_analysis(
                    english_input, current_image, gemini_client
                )
            bot_response = (
                translate_text(analysis, "en", detected_lang)
                if detected_lang != "en" else analysis
            )
            source = f"Gemini Vision ({GEMINI_MODEL})"
            st.session_state.attached_image      = None
            st.session_state.attached_image_name = None

        # ── Task 4: arXiv — FIX 2: extended follow-up keywords ───
        elif is_arxiv_question(english_input) or (
            st.session_state.arxiv_active_paper is not None and any(
                w in english_input.lower()
                for w in ARXIV_FOLLOWUP_KEYWORDS
            )
        ):
            arxiv_response, source = get_arxiv_response(english_input)
            bot_response = arxiv_response

        elif vector_db is not None:
            kb_results = search_knowledge_base(english_input)
            if kb_results:
                best            = kb_results[0]
                bot_response_en = prefix + best["content"]
                bot_response    = (
                    translate_text(bot_response_en, "en", detected_lang)
                    if detected_lang != "en" else bot_response_en
                )
                source = f"Knowledge Base — {best['source']}"
            elif is_medical_question(english_input) and medquad_available:
                common_response = get_common_symptom_response(english_input)
                if common_response:
                    bot_response_en = prefix + common_response
                    bot_response    = (
                        translate_text(bot_response_en, "en", detected_lang)
                        if detected_lang != "en" else bot_response_en
                    )
                    source = "Common Symptom Guide"
                else:
                    best_match, score = get_medical_response(
                        english_input, qa_data, med_vectorizer, med_tfidf
                    )
                    if best_match:
                        bot_response_en = (
                            prefix +
                            f"**Topic:** {best_match['focus']}\n\n" +
                            best_match["answer"]
                        )
                        bot_response = (
                            translate_text(bot_response_en, "en", detected_lang)
                            if detected_lang != "en" else bot_response_en
                        )
                        source = f"MedQuAD ({round(score*100,1)}%)"
                    else:
                        bot_response_en = (
                            prefix +
                            "I couldn't find a reliable medical answer. "
                            "Please consult a doctor."
                        )
                        bot_response = (
                            translate_text(bot_response_en, "en", detected_lang)
                            if detected_lang != "en" else bot_response_en
                        )
            else:
                gym_response = get_gym_response(
                    english_input, model, words, classes, intents
                )
                if gym_response:
                    bot_response_en = prefix + gym_response
                    bot_response    = (
                        translate_text(bot_response_en, "en", detected_lang)
                        if detected_lang != "en" else bot_response_en
                    )
                    source = "Gym Knowledge Base"
                else:
                    if GROQ_AVAILABLE:
                        try:
                            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                            response = groq_client.chat.completions.create(
                                model="llama3-8b-8192",
                                messages=[{"role": "user", "content": english_input}],
                                max_tokens=300,
                            )
                            bot_response_en = response.choices[0].message.content.strip()
                            source = "Groq AI (llama3)"
                        except Exception:
                            bot_response_en = (
                                prefix +
                                "I'm not sure about that. Try: *find papers on [topic]* for CS research."
                            )
                    else:
                        bot_response_en = (
                            prefix +
                            "I'm not sure about that. Try: *find papers on [topic]* for CS research."
                        )

        elif is_medical_question(english_input) and medquad_available:
            common_response = get_common_symptom_response(english_input)
            if common_response:
                bot_response_en = prefix + common_response
                bot_response    = (
                    translate_text(bot_response_en, "en", detected_lang)
                    if detected_lang != "en" else bot_response_en
                )
                source = "Common Symptom Guide"
            else:
                best_match, score = get_medical_response(
                    english_input, qa_data, med_vectorizer, med_tfidf
                )
                if best_match:
                    bot_response_en = (
                        prefix +
                        f"**Topic:** {best_match['focus']}\n\n" +
                        best_match["answer"]
                    )
                    bot_response = (
                        translate_text(bot_response_en, "en", detected_lang)
                        if detected_lang != "en" else bot_response_en
                    )
                    source = f"MedQuAD ({round(score*100,1)}%)"
                else:
                    bot_response_en = (
                        prefix +
                        "I couldn't find a reliable medical answer. "
                        "Please consult a doctor."
                    )
                    bot_response = (
                        translate_text(bot_response_en, "en", detected_lang)
                        if detected_lang != "en" else bot_response_en
                    )

        else:
            gym_response = get_gym_response(
                english_input, model, words, classes, intents
            )
            if gym_response:
                bot_response_en = prefix + gym_response
                bot_response    = (
                    translate_text(bot_response_en, "en", detected_lang)
                    if detected_lang != "en" else bot_response_en
                )
                source = "Gym Knowledge Base"
            else:
                bot_response_en = (
                    prefix +
                    "I'm not sure about that. Ask me about fitness, gym, health, "
                    "or try: *find papers on [topic]* for CS research."
                )
                bot_response = (
                    translate_text(bot_response_en, "en", detected_lang)
                    if detected_lang != "en" else bot_response_en
                )

        st.session_state.main_messages.append({
            "role":    "assistant",
            "content": bot_response,
            "source":  source,
        })

        st.rerun()