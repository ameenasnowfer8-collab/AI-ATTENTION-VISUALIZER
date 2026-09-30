# AI Attention Visualizer

## Project Overview

AI Attention Visualizer is a Streamlit-based application that demonstrates a simple attention mechanism using text extracted from a study-notes image.

The application follows this flow:

Image → OCR → Extract Words → Embeddings → Attention → Word Attention Bars

## Objective

The main objective of this project is to:

- Upload a study-notes image.
- Extract text from the image using OCR.
- Split the extracted text into words.
- Convert words into numerical embeddings.
- Apply a basic scaled dot-product attention mechanism.
- Calculate attention scores for the words.
- Display the attention scores using progress bars.
- Show the word with the highest calculated attention score.

## Technologies Used

- Python
- Streamlit
- Tesseract OCR
- pytesseract
- Sentence Transformers
- NumPy
- Pillow

## Project Structure

AI-Attention-Visualizer/
│
├── app.py
├── ocr.py
├── embedding.py
├── attention.py
├── requirements.txt
└── README.md

## System Architecture

### 1. Image Upload

The user uploads a JPG, JPEG, or PNG study-notes image through the Streamlit interface.

### 2. OCR

Tesseract OCR extracts the text from the uploaded image.

### 3. Word Processing

The extracted text is split into individual words and basic punctuation is removed. The application uses the first 20 words for the attention visualization.

### 4. Embeddings

The `all-MiniLM-L6-v2` Sentence Transformer model converts the words into numerical embeddings.

### 5. Attention

NumPy is used to create Query (Q), Key (K), and Value (V) representations and calculate scaled dot-product attention.

### 6. Visualization

The attention scores are normalized and displayed as Streamlit progress bars. The word with the highest calculated score is also displayed.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
