# 🤖 AI Course Recommender Chatbot

An AI-powered chatbot that recommends suitable online courses based on user queries using semantic search. It leverages the `all-mpnet-base-v2` model for contextual embeddings and Pinecone vector database for efficient similarity-based retrieval.

---

## Project Overview

This chatbot simplifies course discovery by allowing users to enter natural language queries and receive personalized recommendations. It uses advanced NLP techniques to understand user intent and match it to relevant courses from a curated dataset (scraped from [Brainlox](https://brainlox.com)).

---

## Features

-  Semantic understanding of queries using transformer embeddings.
-  Real-time course recommendations from a structured course database.
-  Efficient similarity search using Pinecone.
-  Conversational CLI-based interface (with future Streamlit UI support).
-  Scalable design for expanding datasets or course providers.

---

## Tech Stack

| Component               | Tool/Library                        |
|------------------------|-------------------------------------|
| Programming Language   | Python                              |
| Embedding Model        | all-mpnet-base-v2 (Hugging Face)    |
| Vector Search Engine   | Pinecone                            |
| Web Scraping           | BeautifulSoup                       |
| Data Handling          | Pandas, NumPy                       |
| UI Prototype           | Command Line Interface, Streamlit   |
| IDEs Used              | Jupyter Notebook, VS Code           |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-course-recommender-chatbot.git
cd ai-course-recommender-chatbot

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
