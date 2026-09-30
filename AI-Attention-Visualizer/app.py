import streamlit as st
import numpy as np
from PIL import Image

from ocr import extract_text
from embedding import create_embeddings
from attention import calculate_attention


st.title("AI Attention Visualizer")

st.write("Upload a study-notes image to visualize word attention.")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    # Step 1: OCR
    text = extract_text(image)

    st.subheader("Extracted Text")
    st.write(text)

    # Step 2: Extract words
    words = text.split()

    words = [
        word.strip(".,!?;:\"'()[]{}")
        for word in words
    ]

    words = [word for word in words if word]

    # Keep first 20 words
    words = words[:20]

    if len(words) > 0:

        # Step 3: Create embeddings
        embeddings = create_embeddings(words)

        # Step 4: Calculate attention
        attention_scores = calculate_attention(embeddings)

        # Normalize scores
        min_score = np.min(attention_scores)
        max_score = np.max(attention_scores)

        if max_score != min_score:
            normalized_scores = (
                (attention_scores - min_score)
                / (max_score - min_score)
            )
        else:
            normalized_scores = attention_scores

        # Step 5: Display attention
        st.subheader("Word Attention")

        for word, score in zip(words, normalized_scores):

            st.write(f"**{word}**")

            st.progress(float(score))

        # Highest attention word
        highest_index = np.argmax(attention_scores)

        st.subheader("Highest Attention Word")

        st.success(words[highest_index])

    else:
        st.warning("No readable words were found.")