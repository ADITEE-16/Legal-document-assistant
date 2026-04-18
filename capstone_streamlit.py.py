import streamlit as st
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from datetime import datetime

os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="⚖️",
    layout="centered"
)

@st.cache_resource
def load_resources():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.environ["GROQ_API_KEY"]
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    class SimpleVectorStore:
        def __init__(self):
            self.documents = []
            self.embeddings = []
            self.ids = []
            self.metadatas = []
        def add(self, documents, embeddings, ids, metadatas):
            self.documents.extend(documents)
            self.embeddings.extend(embeddings)
            self.ids.extend(ids)
            self.metadatas.extend(metadatas)
        def count(self):
            return len(self.documents)
        def query(self, query_embeddings, n_results=3):
            query_vec = np.array(query_embeddings[0])
            all_vecs  = np.array(self.embeddings)
            dot    = all_vecs @ query_vec
            norms  = np.linalg.norm(all_vecs, axis=1) * np.linalg.norm(query_vec)
            scores = dot / (norms + 1e-9)
            top_indices = np.argsort(scores)[::-1][:n_results]
            return {
                "documents": [[self.documents[i] for i in top_indices]],
                "metadatas": [[self.metadatas[i] for i in top_indices]],
                "ids":       [[self.ids[i]       for i in top_indices]],
            }

    documents = [
        {"id":"doc_001","topic":"Non-Disclosure Agreement (NDA) Basics","text":"A Non-Disclosure Agreement (NDA) is a legally binding contract that establishes a confidential relationship between parties. Types of NDAs: Unilateral NDA where one party shares confidential information, Bilateral NDA where both parties share information, and Multilateral NDA involving three or more parties. Key clauses include definition of confidential information, obligations of receiving party, exclusions from confidentiality, duration typically 2 to 5 years, and remedies for breach. Violation can result in lawsuits, injunctions, and monetary damages."},
        {"id":"doc_002","topic":"Employment Contract Key Clauses","text":"An employment contract outlines terms and conditions of employment. Essential clauses include Job Title and Description, Compensation and Benefits, Working Hours, Probation Period typically 3 to 6 months, Confidentiality Clause, Non-Compete Clause restricting joining competitors for 6 months to 2 years, Intellectual Property where work belongs to employer, Termination Clause with notice period of 30 to 90 days, Dispute Resolution through arbitration or mediation, and Governing Law."},
        {"id":"doc_003","topic":"Contract Breach and Remedies","text":"A breach of contract occurs when a party fails to fulfill obligations. Types include Material Breach which is significant failure, Minor Breach which is partial failure, Anticipatory Breach where party indicates in advance they won't perform, and Actual Breach on due date. Remedies include Compensatory Damages, Consequential Damages, Liquidated Damages, Specific Performance, Rescission, and Injunction. Statute of limitations is typically 3 to 6 years."},
        {"id":"doc_004","topic":"Intellectual Property Rights Overview","text":"IP types include Copyright protecting creative works for creator's lifetime plus 60 years in India, Trademark protecting brand identifiers renewed every 10 years, Patent protecting inventions for 20 years from filing date after which invention enters public domain, and Trade Secret protecting confidential business information with no expiry date. IP infringement results in civil lawsuits and criminal penalties for willful infringement."},
        {"id":"doc_005","topic":"Arbitration and Dispute Resolution","text":"Arbitration is alternative dispute resolution through a neutral arbitrator. Binding arbitration is final and enforceable. Non-binding arbitration is advisory only. Types of ADR include Negotiation, Mediation where mediator facilitates voluntary settlement, Arbitration with binding decision, and Conciliation. Arbitration clauses specify institution, number of arbitrators, seat, language, and governing law. In India governed by Arbitration and Conciliation Act 1996."},
        {"id":"doc_006","topic":"Sale and Purchase Agreement","text":"A Sale and Purchase Agreement (SPA) outlines terms of a transaction. Key components include Parties, Description of Asset, Purchase Price and payment terms, Representations and Warranties, Conditions Precedent such as regulatory approvals, Indemnification, Closing Date when ownership transfers, Default and Termination clauses, Confidentiality, and Governing Law. Common issues include disputes over warranties and failure to meet conditions precedent."},
        {"id":"doc_007","topic":"Power of Attorney","text":"Power of Attorney (POA) grants an agent authority to act on behalf of the principal. Types include General POA for broad powers, Special or Limited POA for specific tasks, Durable POA that remains valid during incapacitation, Medical POA for healthcare decisions, and Springing POA activated by specific event. Requirements include principal of sound mind, voluntary signing, two witnesses, notarization, and registration for property transactions. POA can be revoked at any time by executing a revocation document."},
        {"id":"doc_008","topic":"Company Incorporation and Corporate Law","text":"In India companies are incorporated under Companies Act 2013. Types include Private Limited Company with minimum 2 directors and maximum 200 shareholders, Public Limited Company with minimum 3 directors, One Person Company for solo entrepreneurs, and Limited Liability Partnership. Steps to incorporate include obtaining DSC, applying for DIN, reserving company name through RUN application, filing SPICe+ form with MOA and AOA, receiving Certificate of Incorporation from ROC, applying for PAN and TAN, and opening bank account."},
        {"id":"doc_009","topic":"Lease and Rental Agreement","text":"A lease agreement governs rental of property. Essential clauses include Parties, Property Description, Lease Term, Rent Amount and due date, Security Deposit typically 2 to 3 months rent, Maintenance Responsibilities, Permitted Use, Subletting terms, Termination Notice of 1 to 3 months, and Lock-in Period. In India lease agreements for 12 months or more must be registered with Sub-Registrar office. Unregistered leases above 12 months are not admissible as evidence in court."},
        {"id":"doc_010","topic":"Legal Due Diligence Process","text":"Legal due diligence is comprehensive investigation before major business transactions. Purpose includes identifying legal risks, verifying ownership, confirming compliance, assessing litigation, and evaluating contracts. Key areas include Corporate Documents review, Contracts and Agreements, Intellectual Property verification, Litigation assessment, Regulatory Compliance, Employment Matters, and Financial Obligations. Findings are summarized in a due diligence report identifying red flags and deal breakers which directly impact transaction price."}
    ]

    collection = SimpleVectorStore()
    for doc in documents:
        embedding = embedder.encode(doc["text"]).tolist()
        collection.add(
            documents=[doc["text"]],
            embeddings=[embedding],
            ids=[doc["id"]],
            metadatas=[{"topic": doc["topic"]}]
        )

    class LegalState(TypedDict):
        question     : str
        messages     : List[str]
        route        : str
        retrieved    : str
        sources      : List[str]
        tool_result  : str
        answer       : str
        faithfulness : float
        eval_retries : int
        user_name    : str

    def memory_node(state):
        messages  = state.get("messages", [])
        question  = state["question"]
        user_name = state.get("user_name", "")
        if "my name is" in question.lower():
            parts = question.lower().split("my name is")
            if len(parts) > 1:
                user_name = parts[1].strip().split()[0].capitalize()
        messages.append(f"User: {question}")
        messages = messages[-6:]
        return {**state, "messages": messages, "user_name": user_name}

    def router_node(state):
        question = state["question"]
        history  = "\n".join(state.get("messages", []))
        prompt = f"""Route the legal query to ONE word only:
- retrieve  → legal questions about NDA, contracts, IP, arbitration, lease, POA, company law, due diligence, breach
- tool      → current date or time
- skip      → casual greeting or thank you
History: {history}
Question: {question}
Reply (retrieve / tool / skip):"""
        response = llm.invoke(prompt)
        route = response.content.strip().lower().split()[0]
        if route not in ["retrieve", "tool", "skip"]:
            route = "retrieve"
        return {**state, "route": route}

    def retrieval_node(state):
        query_embedding = embedder.encode(state["question"]).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        chunks  = []
        sources = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(f"[{meta['topic']}]\n{doc}")
            sources.append(meta["topic"])
        return {**state, "retrieved": "\n\n".join(chunks), "sources": sources}

    def skip_node(state):
        return {**state, "retrieved": "", "sources": []}

    def tool_node(state):
        question = state["question"].lower()
        try:
            now = datetime.now()
            if "time" in question:
                result = f"The current time is {now.strftime('%I:%M %p')}."
            elif "date" in question:
                result = f"Today's date is {now.strftime('%A, %d %B %Y')}."
            else:
                result = f"Today is {now.strftime('%A, %d %B %Y')} and the time is {now.strftime('%I:%M %p')}."
        except:
            result = "Sorry, I could not fetch the current date/time."
        return {**state, "tool_result": result}

    def answer_node(state):
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        history      = "\n".join(state.get("messages", []))
        user_name    = state.get("user_name", "")
        eval_retries = state.get("eval_retries", 0)
        name_str     = f"The user's name is {user_name}." if user_name else ""
        context      = f"LEGAL KNOWLEDGE BASE:\n{retrieved}" if retrieved else f"TOOL RESULT:\n{tool_result}"
        retry_note   = "Be more precise and faithful to the context." if eval_retries > 0 else ""
        prompt = f"""You are a professional legal document assistant for paralegals and lawyers.
{name_str}
RULES:
1. Answer ONLY from the information below.
2. If unsure say: I don't have that information. Please consult a qualified lawyer.
3. Never give personal legal advice for specific cases.
4. Be professional and cite relevant legal concepts.
{retry_note}
{context}
History: {history}
Question: {question}
Answer:"""
        response = llm.invoke(prompt)
        return {**state, "answer": response.content.strip(), "tool_result": ""}

    def eval_node(state):
        answer       = state.get("answer", "")
        retrieved    = state.get("retrieved", "")
        eval_retries = state.get("eval_retries", 0)
        if not retrieved:
            return {**state, "faithfulness": 1.0, "eval_retries": eval_retries}
        prompt = f"""Rate faithfulness 0.0 to 1.0.
Context: {retrieved}
Answer: {answer}
Reply with number only:"""
        try:
            response     = llm.invoke(prompt)
            faithfulness = float(response.content.strip().split()[0])
            faithfulness = max(0.0, min(1.0, faithfulness))
        except:
            faithfulness = 0.8
        return {**state, "faithfulness": faithfulness, "eval_retries": eval_retries + 1}

    def save_node(state):
        messages = state.get("messages", [])
        messages.append(f"Assistant: {state.get('answer', '')}")
        return {**state, "messages": messages[-6:]}

    def route_decision(state):
        r = state.get("route", "retrieve")
        return r if r in ["retrieve", "tool", "skip"] else "retrieve"

    def eval_decision(state):
        return "save" if state.get("faithfulness", 1.0) >= 0.7 or state.get("eval_retries", 0) >= 2 else "answer"

    graph = StateGraph(LegalState)
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)
    graph.set_entry_point("memory")
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)
    graph.add_conditional_edges("router", route_decision,
                                {"retrieve":"retrieve","skip":"skip","tool":"tool"})
    graph.add_conditional_edges("eval", eval_decision,
                                {"save":"save","answer":"answer"})
    app = graph.compile(checkpointer=MemorySaver())
    return app, LegalState

# Session state
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "thread_id"    not in st.session_state:
    st.session_state.thread_id    = "legal_user_001"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

app, LegalState = load_resources()

# Sidebar
with st.sidebar:
    st.title("⚖️ Legal Assistant")
    st.markdown("**Your AI-powered Legal Document Helper**")
    st.markdown("---")
    st.markdown("**Topics I can help with:**")
    st.markdown("""
- 📄 Non-Disclosure Agreements
- 💼 Employment Contracts
- ⚠️ Contract Breach & Remedies
- 🎨 Intellectual Property Rights
- 🤝 Arbitration & Dispute Resolution
- 🏠 Sale & Purchase Agreements
- 📋 Power of Attorney
- 🏢 Company Incorporation
- 🔑 Lease & Rental Agreements
- 🔍 Legal Due Diligence
    """)
    st.markdown("---")
    st.warning("⚠️ This assistant provides general legal information only. Always consult a qualified lawyer for specific legal advice.")
    st.markdown("---")
    if st.button("🔄 New Conversation"):
        st.session_state.messages     = []
        st.session_state.thread_id    = f"legal_{int(datetime.now().timestamp())}"
        st.session_state.chat_history = []
        st.rerun()

# Main UI
st.title("⚖️ Legal Document Assistant")
st.caption("AI-powered assistant for paralegals and legal professionals")
st.markdown("---")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your legal question... e.g. What is an NDA?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state = {
                "question"    : prompt,
                "messages"    : st.session_state.messages,
                "route"       : "",
                "retrieved"   : "",
                "sources"     : [],
                "tool_result" : "",
                "answer"      : "",
                "faithfulness": 0.0,
                "eval_retries": 0,
                "user_name"   : ""
            }
            result = app.invoke(initial_state, config=config)
            answer = result["answer"]
            st.session_state.messages = result["messages"]
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})