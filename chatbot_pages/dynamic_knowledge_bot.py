import streamlit as st
import os

from services.kb_updater import (
    build_vector_database
)

from services.kb_search import (
    search_knowledge_base
)


def show():



    st.title("📚 Dynamic Knowledge Bot")

    st.markdown(
        """
        Upload **TXT, PDF, or DOCX** documents
        to dynamically expand the chatbot knowledge base.
        """
    )



    uploaded_file = st.file_uploader(

        "Upload knowledge documents",

        type=["txt", "pdf", "docx"]
    )

    if uploaded_file is not None:

        os.makedirs(
            "knowledge_base/documents",
            exist_ok=True
        )

        save_path = os.path.join(

            "knowledge_base/documents",

            uploaded_file.name
        )

        with open(save_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        st.success(
            f"{uploaded_file.name} uploaded successfully!"
        )


    if st.button("🔄 Update Knowledge Base"):

        with st.spinner(
            "Updating vector database..."
        ):

            try:

                total_chunks = build_vector_database()

                st.success(

                    f"""
                    Knowledge base updated successfully!

                    Indexed {total_chunks} chunks.
                    """
                )

            except Exception as e:

                st.error(
                    f"Knowledge base update failed: {e}"
                )



    query = st.text_input(
        "Ask a question from uploaded documents:"
    )

    if query:

        try:

            results = search_knowledge_base(query)

            st.subheader("Top Results")



            if not results:

                st.warning(
                    "No relevant information found."
                )



            else:

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### Result {i}"
                    )

                    # Safe dictionary access
                    source = result.get(
                        "source",
                        "Unknown"
                    )

                    content = result.get(
                        "content",
                        "No content found."
                    )

                    st.markdown(
                        f"**Source:** {source}"
                    )

                    st.write(content)

                    st.markdown("---")

        except Exception as e:

            st.error(
                f"Search failed: {e}"
            )