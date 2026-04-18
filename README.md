# ⚖️ Legal Document Assistant

An AI-powered legal document assistant built using LangGraph, ChromaDB RAG, 
and Streamlit for the Agentic AI Capstone Project 2026.

## What it does
- Answers legal questions from a knowledge base of 10 legal documents
- Topics covered: NDA, Employment Contracts, Contract Breach, IP Rights,
  Arbitration, Sale & Purchase Agreement, Power of Attorney, 
  Company Incorporation, Lease Agreements, Legal Due Diligence
- Remembers conversation context using MemorySaver + thread_id
- Self-reflection evaluation node for answer quality
- Datetime tool for date and deadline queries
- Streamlit UI for easy interaction

## Tech Stack
- LangGraph — agent graph with 8 nodes
- Groq (llama-3.1-8b-instant) — LLM
- SentenceTransformer (all-MiniLM-L6-v2) — embeddings
- SimpleVectorStore — vector database
- Streamlit — user interface
- Python 3.11

## How to run

### Install dependencies
pip install streamlit langchain-groq langgraph sentence-transformers numpy

### Add your Groq API key
Open legal_streamlit.py and replace the api_key value:
api_key="your_groq_api_key_here"
Get a free API key from https://console.groq.com

### Run the app
streamlit run legal_streamlit.py

## Project Structure
- legal_streamlit.py — Streamlit UI and complete agent code
- legal_assistant.ipynb — Development notebook with all 8 parts

## Agent Architecture
User Question
→ memory_node — stores conversation history
→ router_node — decides retrieve / tool / skip
→ retrieval_node — fetches relevant legal documents
→ answer_node — generates grounded answer
→ eval_node — checks faithfulness score
→ save_node — saves to conversation history

## Evaluation Scores (Baseline)
 -Faithfulness      : 0.80
 -Answer Relevancy  : 0.80
 -Context Precision : 0.80

## Author
Name: ADITEE TRIPATHY
Roll Number: 23052047
Batch: CSE 2027
Course: Agentic AI — 2026
