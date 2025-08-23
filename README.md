# 🤖 Customer Support Chatbots 

This repository contains two chatbot implementations developed as part of Task 3 of the Future Interns internship program. Both bots aim to handle customer queries intelligently and efficiently, but differ in architecture and complexity.

---

## 📌 Overview

### 1. Basic FAQ Bot – Dialogflow Essentials + Telegram
A quick and lightweight customer support chatbot built using:
- Dialogflow Essentials for intent classification and response generation
- Telegram Bot API for deployment and interaction

🔗 [Live Bot Link](http://bit.ly/4fj9W2M)

#### Features:
- Intent-based matching via Dialogflow
- Predefined training phrases and responses
- Fast and simple deployment
- Good for static or rule-based FAQs

---

### 2. RAG-Based Bot – Sentence Transformer + FAISS + Groq API + Streamlit
An advanced Retrieval-Augmented Generation (RAG) chatbot with a custom chat-like UI:
- Retrieves relevant answers from a 27K customer support dataset
- Uses SentenceTransformer (all-MiniLM-L6-v2) for semantic embeddings
- Stores vectors in a FAISS index for fast retrieval
- Integrates with Groq LLM (Gemma-2-9B-IT) for high-quality responses
- Provides direct answers when confidence is high, otherwise uses RAG + LLM
- Fully deployed on Streamlit with a modern chat UI


#### Features:
- Context-aware support system powered by embeddings + Groq API
- Direct response mechanism when similarity score ≥ 0.80
- RAG pipeline for more complex queries
- Clean chat-like UI with user/bot message bubbles and autoscroll
- Background customization with image + CSS styling
- Session-based chat history

---

## 🚀 Getting Started

### 🔧 Requirements

Install dependencies:
``` bash
pip install -r requirements.txt
```

### ⚙️ Environment Setup
You need a Groq API key. Add it inside .streamlit/secrets.toml:
``` bash
GROQ_API_KEY = "your_api_key_here"
```
## Folder Structure

``` bash
├── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv  
├── build_index.py                      
├── chatbot.py                
├── embeddings.npy            
├── faiss.index               
├── requirements.txt  
├── README.md  
├── bg3.jpeg                  
└── diagflow/                 
```

## ✅ Sample Queries to Try

- "I want to return an item"
- "How can I reset my password?"
- "My order hasn’t arrived yet"
- "Cancel my subscription"
- "Where do I find my invoice?"
- "Any student discount available?"



## 📢 Connect


🔗 [Let's connect on LinkedIn](https://www.linkedin.com/in/sugan2111/)


