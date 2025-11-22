import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List
from operator import itemgetter

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ✅ STANDARD IMPORTS (Matches your requirements.txt)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration & basic checks
# -------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is missing!")
    # raise RuntimeError("Please set GEMINI_API_KEY")

INDEX_DIR = os.environ.get("INDEX_DIR", "faiss_index")

if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

@dataclass
class RagConfig:
    language: str = "en"

# -------------------------------------------------------------------
# Load FAISS index
# -------------------------------------------------------------------

def load_vector_index():
    """Load the FAISS index built by build_index.py."""
    logger.info(f"Loading index from {INDEX_DIR}...")
    
    # ✅ FIX: Must match build_index.py (text-embedding-004)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    if not os.path.isdir(INDEX_DIR):
        logger.critical(f"Index directory '{INDEX_DIR}' not found!")
        return None

    try:
        index = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            index_name="index",
            allow_dangerous_deserialization=True,
        )
        logger.info("✅ Index loaded successfully.")
        return index
    except Exception as e:
        logger.critical(f"Failed to load index: {e}")
        return None

vector_index = load_vector_index()
if vector_index:
    retriever = vector_index.as_retriever(search_kwargs={"k": 10})
else:
    retriever = None

# -------------------------------------------------------------------
# Helper: system prompts per language
# -------------------------------------------------------------------

def get_system_prompt(language: str) -> str:
    if language == "ar":
        return (
            "أنت مساعد مكتب التخطيط والجودة الأكاديمية بجامعة قطر.\n"
            "اعتمد أولاً على الوثائق المتاحة في سياقك.\n"
            "السياق:\n{context}\n\n"
            "تاريخ المحادثة:\n{chat_history}\n\n"
            "السؤال:\n{input}"
        )
    elif language == "fr":
        return (
            "Vous êtes l’assistant du Bureau APQA de l’Université du Qatar.\n"
            "Appuyez-vous d’abord sur les documents fournis.\n"
            "Contexte:\n{context}\n\n"
            "Historique:\n{chat_history}\n\n"
            "Question:\n{input}"
        )
    else:
        return (
            "You are the APQA Assistant at Qatar University.\n"
            "Rely first on the official documents provided.\n"
            "Context:\n{context}\n\n"
            "History:\n{chat_history}\n\n"
            "Question:\n{input}"
        )

# -------------------------------------------------------------------
# Build RAG chain (Modern LCEL)
# -------------------------------------------------------------------

def build_rag_chain(config: RagConfig):
    # ✅ FIX: Using gemini-flash-latest
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2,
        max_output_tokens=1024,
    )

    system_prompt_text = get_system_prompt(config.language)
    
    # Modern ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "{input}"),
    ])

    if not retriever:
        return None

    # ✅ MODERN CHAIN CONSTRUCTION
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Wrapper to handle chat history string formatting
    def chain(inputs: Dict[str, Any]) -> str:
        response = rag_chain.invoke({
            "input": inputs["input"],
            "chat_history": inputs.get("chat_history", [])
        })
        return response['answer']

    return chain

rag_chain_en = build_rag_chain(RagConfig(language="en"))
rag_chain_ar = build_rag_chain(RagConfig(language="ar"))
rag_chain_fr = build_rag_chain(RagConfig(language="fr"))

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

chat_history: List[str] = []

@app.route("/")
def index() -> Any:
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        language = (data.get("language") or "en").lower()

        if not message:
            return jsonify({"error": "Message is required"}), 400

        if not rag_chain_en:
             return jsonify({"reply": "System is starting up or index failed to load. Please check server logs."}), 503

        if language == "ar":
            chain = rag_chain_ar
        elif language == "fr":
            chain = rag_chain_fr
        else:
            chain = rag_chain_en

        # Simple history management
        history_text = "\n".join(chat_history[-6:])

        logger.info(f"Processing message ({language}): {message[:50]}...")
        answer = chain({
            "input": message,
            "chat_history": history_text,
        })
        
        chat_history.append(f"User: {message}")
        chat_history.append(f"Assistant: {answer}")

        return jsonify({"reply": answer})

    except Exception as e:
        logger.error(f"❌ Error processing request: {e}", exc_info=True)
        return jsonify({"reply": "Sorry, something went wrong. Please try again."}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)