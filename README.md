# RAG-multimodal-pipeline
---

## 🧰 Tech Stack

- Python 3.10+
- [Google Gemini API](https://makersuite.google.com/)
- [SigLIP via HuggingFace Transformers](https://huggingface.co/google/siglip-so400m-patch14-384)
- FAISS for similarity search
- PDF2Image & PyMuPDF for PDF parsing
- PIL for image processing

---

## ⚙️ Setup Instructions (Colab)

1. **Open the Colab notebook**: [🔗 Link to notebook]
2. **Install dependencies** in the first cell.
3. **Upload your files** (PDF or ZIP containing images and text).
4. **Set your Gemini API key** in cell 2.
5. **Run all cells** from top to bottom.
6. Use the `ask()` function to query your document:
   ```python
   ask("What is this document about?")
