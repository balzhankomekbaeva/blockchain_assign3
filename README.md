# 🇰🇿 Kazakhstan Constitution Assistant

A free-tier compatible AI assistant that answers questions based strictly on the Constitution of the Republic of Kazakhstan and optionally supports your own uploaded documents.

---

## 🔧 Usage

1. **Install dependencies**  
   Ensure you have Python installed. Then install requirements:
   ```bash
   pip install -r requirements.txt
2.Set up environment variables
Create a .env file with your HuggingFace API Token:
  ```bash
HUGGINGFACEHUB_API_TOKEN=your_token_here
```
3.Run the app
  ```bash
streamlit run app.py
```
Using the app

  Click "🔄 Load Constitution" to fetch the Constitution from the official government website.

  Upload additional .pdf, .docx, or .txt files if desired.

  Ask questions in the chat box.

  The assistant will respond with short answers strictly based on the documents.
📸 Demo Screenshots


🧠 Examples
Q: What is the official language of the Republic of Kazakhstan?
A: The official language is Kazakh.

Q: Can the President dissolve Parliament?
A: Yes, under certain constitutional conditions.

Q: What does the Constitution say about internet freedom?
A: Not addressed in the Constitution.
