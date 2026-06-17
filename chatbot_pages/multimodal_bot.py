
import streamlit as st


def show():


    import os
    import io
    import datetime
    import requests
    from PIL import Image
    from dotenv import load_dotenv

    # ── Gemini SDK ───────────────────────────────────────────────
    try:
        from google import genai
        from google.genai import types
        GEMINI_AVAILABLE = True
    except ImportError:
        try:
            import google.generativeai as genai_old
            GEMINI_AVAILABLE = True
            NEW_SDK = False
        except ImportError:
            GEMINI_AVAILABLE = False
        else:
            NEW_SDK = False
    else:
        NEW_SDK = True

    # ── Groq SDK ─────────────────────────────────────────────────
    try:
        from groq import Groq
        GROQ_AVAILABLE = True
    except ImportError:
        GROQ_AVAILABLE = False



    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

    if not GOOGLE_API_KEY and not GROQ_API_KEY:
        st.error(
            "❌ No API keys found!\n\n"
            "Add to your `.env` file:\n"
            "```\n"
            "GOOGLE_API_KEY=your_gemini_key\n"
            "GROQ_API_KEY=your_groq_key\n"
            "```"
        )
        return

    
    gemini_client = None

    if GOOGLE_API_KEY and GEMINI_AVAILABLE:
        try:
            if NEW_SDK:
                gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
            else:
                genai_old.configure(api_key=GOOGLE_API_KEY)
                gemini_client = "old_sdk"
        except Exception:
            gemini_client = None

    groq_client = None

    if GROQ_API_KEY and GROQ_AVAILABLE:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
        except Exception:
            groq_client = None


    GEMINI_MODEL = "gemini-2.0-flash"


    IMAGE_GEN_KEYWORDS = [
        "generate image", "create image", "make image",
        "generate a picture", "create a picture",
        "generate photo", "create photo", "draw", "draw me",
        "paint", "imagine", "visualize",
        "generate an image of", "create an image of",
        "show me a picture of", "make an image of",
        "generate a", "create a", "make a",   # ✅ catches "generate a lion image"
    ]

    import re
    IMAGE_GEN_PATTERNS = [
        r"generate\s+\w+\s+image",   # "generate a lion image"
        r"generate\s+an?\s+image",   # "generate an image"
        r"create\s+\w+\s+image",     # "create a cat image"
        r"make\s+\w+\s+image",       # "make a dog image"
        r"generate\s+image\s+of",    # "generate image of"
        r"(draw|paint|sketch)\s+",   # "draw a sunset"
        r"show\s+me\s+.*(image|picture|photo)",
    ]

    CREATIVE_KEYWORDS = [
        "write a poem", "write a story", "write an essay",
        "write a song", "write a joke", "creative writing",
        "tell me a story", "compose a", "write me a",
    ]

    def is_image_gen(text):
        t = text.lower()
        if any(kw in t for kw in IMAGE_GEN_KEYWORDS):
            return True
        if any(re.search(p, t) for p in IMAGE_GEN_PATTERNS):
            return True
        return False

    def is_creative(text):
        return any(kw in text.lower() for kw in CREATIVE_KEYWORDS)

    IMAGE_QUESTION_KEYWORDS = [
        "describe", "what is", "what's in", "what are", "what do you see",
        "analyze", "analyse", "explain", "identify", "recognize", "recognise",
        "tell me about", "look at", "read", "text in", "words in",
        "color", "colour", "object", "person", "animal", "background",
        "how many", "where is", "what kind", "type of", "style of",
        "scene", "image", "picture", "photo", "this", "shown", "visible",
    ]

    def is_image_question(text):
        """Returns True only if the user is asking about the uploaded image."""
        t = text.lower()
        return any(kw in t for kw in IMAGE_QUESTION_KEYWORDS)

    def extract_image_prompt(text):
        t = text.lower()
        m = re.search(
            r"(?:generate|create|make|draw|paint|sketch)\s+(?:a\s+|an\s+)?(.*?)(?:\s+image|\s+picture|\s+photo)?$",
            t
        )
        if m:
            subject = m.group(1).strip(" .:,of")
            if subject:
                return subject
        for phrase in sorted(IMAGE_GEN_KEYWORDS, key=len, reverse=True):
            if phrase in t:
                idx  = t.find(phrase)
                rest = text[idx + len(phrase):].strip(" .:,of")
                return rest if rest else text
        return text



    def groq_text(prompt, history, system_prompt=None):

        if not groq_client:
            return "❌ Groq not available. Please add GROQ_API_KEY to .env"

        try:

            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer clearly and concisely."
                })

            # Add chat history
            for msg in history[-6:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role":    msg["role"],
                        "content": msg["content"]
                    })

            messages.append({"role": "user", "content": prompt})

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",   # ✅ FIXED: updated from decommissioned llama3-70b-8192
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"❌ Groq error: {str(e)}"


    def gemini_image_analysis(prompt, image, history):

        if not gemini_client:
            return (
                "⚠️ Gemini not available for image analysis.\n\n"
                "Please add GOOGLE_API_KEY to your .env file."
            )

        system_context = (
            "You are a helpful multimodal AI assistant. "
            "Analyze the provided image carefully and "
            "respond to the user's question in detail."
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
                    "⚠️ **Gemini free tier quota reached.**\n\n"
                    "Please wait a minute and try again, "
                    "or wait until tomorrow for the daily limit to reset.\n\n"
                    "*Note: Image analysis requires Gemini API.*"
                )
            return f"❌ Image analysis error: {e}"

    def gemini_describe_image(image):

        prompt = (
            "Provide a comprehensive description of this image:\n"
            "1. **Main subject** — What is the primary focus?\n"
            "2. **Objects & people** — What else is visible?\n"
            "3. **Colors & style** — Describe the visual style\n"
            "4. **Setting & context** — Where/when is this?\n"
            "5. **Text** — Any visible text?\n"
            "6. **Mood & theme** — What feeling does it convey?"
        )

        return gemini_image_analysis(prompt, image, [])



    def generate_image(prompt, history):

        # Try Imagen via Gemini
        if gemini_client and NEW_SDK and gemini_client != "old_sdk":

            for model_id in [
                "imagen-3.0-generate-001",
                "imagen-3.0-fast-generate-001"
            ]:
                try:
                    result = gemini_client.models.generate_images(
                        model=model_id,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(number_of_images=1),
                    )
                    if result.generated_images:
                        img = result.generated_images[0].image
                        pil_img = Image.open(io.BytesIO(img.image_bytes))
                        return pil_img, f"Imagen ({model_id})", None
                except Exception:
                    continue

        desc = groq_text(
            f"Describe in vivid detail what an AI-generated "
            f"image of '{prompt}' would look like. Cover colors, "
            f"composition, lighting, style, and mood. "
            f"Start with '🎨 **Generated Image Description:**'",
            history,
            system_prompt=(
                "You are a creative AI that vividly describes "
                "images as if you generated them."
            )
        )

        return None, "Groq (Llama 3) Description", desc



    def palm_text(prompt, history):

        if GOOGLE_API_KEY:
            try:
                url = (
                    "https://generativelanguage.googleapis.com"
                    "/v1beta/models/text-bison-001:generateText"
                    f"?key={GOOGLE_API_KEY}"
                )
                resp = requests.post(
                    url,
                    json={
                        "prompt": {"text": prompt},
                        "temperature": 0.75,
                        "maxOutputTokens": 512
                    },
                    timeout=20
                )
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    if candidates:
                        out = candidates[0].get("output", "").strip()
                        if out:
                            return out, "Google PaLM (text-bison)"
            except Exception:
                pass

        result = groq_text(
            prompt,
            history,
            system_prompt=(
                "You are a creative writing AI. "
                "Respond with creativity, style, and flair."
            )
        )

        return result, "Groq Llama 3 (creative)"



    def build_export(messages):

        lines = [f"Multimodal Chat — {datetime.datetime.now()}\n"]

        for msg in messages:

            role  = "You" if msg["role"] == "user" else "Bot"
            model = (
                f" [{msg.get('model_used', '')}]"
                if msg.get("model_used") else ""
            )

            lines.append(f"{role}{model}: {msg['content']}")

            if msg.get("has_image"):
                lines.append("  [Image attached]")

            if msg.get("generated_image"):
                lines.append("  [Generated image]")

            lines.append("")

        return "\n".join(lines)



    st.title("🎨 Multimodal Chatbot (Task 5)")

    st.markdown(
        "Powered by **Google Gemini** (images) + "
        "**Groq Llama 3** (text) + **Google PaLM** (creative)\n\n"
        "Upload images · Analyze images · Generate images · "
        "Creative writing"
    )

    st.markdown("---")

    # ── Session state ────────────────────────────────────────────

    for key, default in {
        "multimodal_messages": [],
        "current_image":       None,
        "image_name":          None,
        "show_download":       False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


    with st.sidebar:

        st.markdown("---")
        st.header("🖼️ Image Upload")

        uploaded = st.file_uploader(
            "Upload image to analyze",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded:

            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption=uploaded.name, use_container_width=True)
            st.session_state.current_image = img
            st.session_state.image_name    = uploaded.name
            st.success(f"✅ {uploaded.name} loaded!")

            if st.button("🔍 Auto Describe", use_container_width=True):
                with st.spinner("Gemini is analyzing..."):
                    desc = gemini_describe_image(img)

                st.session_state.multimodal_messages.append({
                    "role":      "user",
                    "content":   f"[Uploaded: {uploaded.name}] Describe this image.",
                    "has_image": True
                })

                st.session_state.multimodal_messages.append({
                    "role":       "assistant",
                    "content":    desc,
                    "model_used": f"Gemini {GEMINI_MODEL}"
                })

                st.rerun()

        elif st.session_state.current_image:

            st.image(
                st.session_state.current_image,
                caption=st.session_state.image_name,
                use_container_width=True
            )
            st.info(f"📌 Active: {st.session_state.image_name}")

            if st.button("🗑️ Remove Image", use_container_width=True):
                st.session_state.current_image = None
                st.session_state.image_name    = None
                st.rerun()

        st.markdown("---")
        st.markdown("### 🤖 Active Models")
        st.markdown(
            f"- 🔵 **Gemini {GEMINI_MODEL}** — Image analysis\n"
            "- 🎨 **Imagen 3** — Image generation\n"
            "- 🟢 **Groq Llama 3.3** — Text generation\n"
            "- 🟠 **Google PaLM** — Creative writing"
        )

        st.markdown("---")
        st.markdown("### 💡 Try these")
        st.markdown(
            "**Image analysis:**\n"
            "- What is in this image?\n"
            "- Describe the colors\n\n"
            "**Image generation:**\n"
            "- Generate an image of a sunset\n"
            "- Draw a futuristic city\n\n"
            "**Creative (PaLM/Groq):**\n"
            "- Write a poem about AI\n"
            "- Write a short story\n\n"
            "**General text (Groq):**\n"
            "- Explain quantum computing\n"
            "- What is machine learning?"
        )


    c1, c2 = st.columns([1, 5])

    with c1:
        if st.button("🗑️ Clear"):
            st.session_state.multimodal_messages = []
            st.rerun()

    with c2:
        if st.button("💾 Save Chat"):
            st.session_state.show_download = True

    if (
        st.session_state.show_download
        and st.session_state.multimodal_messages
    ):
        st.download_button(
            "📥 Download Chat",
            data=build_export(st.session_state.multimodal_messages),
            file_name="multimodal_chat.txt",
            mime="text/plain",
        )

    # ── Status Bar ───────────────────────────────────────────────

    if st.session_state.current_image:
        st.info(
            f"🖼️ Active image: **{st.session_state.image_name}** — "
            f"Using **Gemini** for visual analysis."
        )
    else:
        st.info(
            "💬 Text mode · **Groq Llama 3.3** ready · "
            "Upload an image or try *'generate an image of...'*"
        )

    st.markdown("---")


    if not st.session_state.multimodal_messages:

        with st.chat_message("assistant"):
            st.markdown(
                "👋 Hello! I'm your **Multimodal AI Assistant**!\n\n"
                "**I can:**\n"
                "- 🖼️ **Analyze images** — upload any image "
                "and ask questions (powered by Gemini)\n"
                "- 🎨 **Generate images** — say "
                "*'generate an image of...'*\n"
                "- ✍️ **Creative writing** — say "
                "*'write a poem about...'* (powered by PaLM/Groq)\n"
                "- 💬 **Answer anything** — powered by Groq Llama 3.3\n\n"
                "Upload an image from the sidebar or start chatting!"
            )


    for msg in st.session_state.multimodal_messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg.get("generated_image"):
                try:
                    st.image(
                        msg["generated_image"],
                        caption="🎨 AI Generated Image",
                        use_container_width=True
                    )
                    buf = io.BytesIO()
                    msg["generated_image"].save(buf, format="PNG")
                    st.download_button(
                        "📥 Download Image",
                        data=buf.getvalue(),
                        file_name="generated.png",
                        mime="image/png",
                        key=f"dl_{id(msg)}",
                    )
                except Exception:
                    pass

            badges = []
            if msg.get("has_image"):
                badges.append("📎 Image attached")
            if msg.get("model_used"):
                badges.append(f"🤖 {msg['model_used']}")

            if badges:
                st.markdown(
                    f"<small style='color:gray'>{' · '.join(badges)}</small>",
                    unsafe_allow_html=True
                )


    user_input = st.chat_input(
        "Ask anything, analyze an image, or say 'generate an image of...'"
    )

    if user_input:

        user_input = user_input.strip()
        has_image  = st.session_state.current_image is not None

        st.session_state.multimodal_messages.append({
            "role":      "user",
            "content":   user_input,
            "has_image": has_image
        })

        history         = st.session_state.multimodal_messages[:-1]
        bot_response    = ""
        model_used      = ""
        generated_image = None

        if is_image_gen(user_input):

            img_prompt = extract_image_prompt(user_input)

            with st.spinner(f"🎨 Generating '{img_prompt}'..."):
                gen_img, gen_src, gen_txt = generate_image(img_prompt, history)

            if gen_img:
                bot_response    = f"🎨 Here's your generated image of **{img_prompt}**!"
                model_used      = gen_src
                generated_image = gen_img
            else:
                bot_response = gen_txt or f"❌ Could not generate: {img_prompt}"
                model_used   = gen_src

        elif is_creative(user_input):

            with st.spinner("✍️ Writing creatively..."):
                bot_response, model_used = palm_text(user_input, history)

        elif has_image and is_image_question(user_input):

            with st.spinner("🔵 Gemini analyzing image..."):
                bot_response = gemini_image_analysis(
                    user_input,
                    st.session_state.current_image,
                    history
                )

            model_used = f"Gemini {GEMINI_MODEL}"

        else:

            with st.spinner("🟢 Groq thinking..."):
                bot_response = groq_text(user_input, history)

            model_used = "Groq Llama 3.3"

        st.session_state.multimodal_messages.append({
            "role":            "assistant",
            "content":         bot_response,
            "model_used":      model_used,
            "generated_image": generated_image,
        })

        st.rerun()