import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
# Import from langchain_classic instead of langchain.chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration & basic checks
# -------------------------------------------------------------------

# Use GOOGLE_API_KEY from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is missing! Please set it in Render env vars.")
else:
    logger.info(f"API Key detected (prefix): {GOOGLE_API_KEY[:8]}*****")

INDEX_DIR = os.environ.get("INDEX_DIR", "faiss_index")
logger.info(f"Using INDEX_DIR = {INDEX_DIR}")

@dataclass
class RagConfig:
    language: str = "en"

# -------------------------------------------------------------------
# Load FAISS index
# -------------------------------------------------------------------

def load_vector_index():
    """Load the FAISS index built by build_index.py."""
    logger.info(f"Loading FAISS index from '{INDEX_DIR}' ...")

    if not GOOGLE_API_KEY:
        logger.critical("Cannot load index: GOOGLE_API_KEY is not set!")
        return None

    # Explicitly pass the API key to avoid DefaultCredentialsError
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

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
        logger.info("✅ FAISS index loaded successfully.")
        return index
    except Exception as e:
        logger.critical(f"❌ Failed to load FAISS index: {e}", exc_info=True)
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
    system_prompt = (
        "You represent Qatar University's Office of Academic Planning & Quality Assurance (APQA). "
        "Answer questions the way a knowledgeable APQA staff member would respond to faculty or programs. "
        "Use clear, professional, and friendly language. Do not sound robotic.\n\n"
        "LANGUAGE CONTROL\n"
        "- You receive a parameter called 'language'.\n"
        "- If language == 'ar', answer fully in Modern Standard Arabic.\n"
        "- If language == 'en', answer fully in English.\n"
        "- Do not mix languages unless the user explicitly asks you to.\n\n"
        "STYLE AND TONE\n"
        "- Speak directly, as if you are part of APQA (e.g., \"Currently, the OAS supports...\").\n"
        "- Do NOT mention words like \"context\", \"documents\", \"manuals\", or \"PDF\" in your answer.\n"
        "- Avoid phrases such as \"Based on the provided context\" or \"The context states\". "
        "Instead, state the information directly, as guidance from APQA.\n"
        "- When helpful, you may use short bullet points, but keep the explanation concise and human-sounding.\n\n"
        "POLICY AND GROUNDEDNESS\n"
        "- Your knowledge comes ONLY from the internal APQA manuals, rubrics, and guidelines provided below.\n"
        "- When information is found in the reference materials, explain it fully and comprehensively.\n"
        "- ONLY if something is truly NOT described in the reference information, say: "
        "\"Our current APQA documents do not specify this point\" or "
        "\"This is not explicitly detailed in the available guidelines.\" "
        "You may then offer a neutral, practical suggestion.\n"
        "- Do not invent new policies.\n"
        "- Normal reasoning is allowed (e.g., counting the number of outcomes in a list).\n\n"
        "SPECIAL HANDLING FOR PEOs / PLOs / STUDENT OUTCOMES / PIs\n"
        "- When the user asks about PEOs, PLOs, Student Outcomes (SOs), or Performance Indicators (PIs):\n"
        "  * Look for numbered or bullet lists that define them.\n"
        "  * State explicitly how many there are (e.g., \"Engineering BS programs have seven outcomes\").\n"
        "  * List each item with its wording, as fully as it appears in the information you see.\n"
        "  * If the list appears incomplete, say that the remaining items do not appear in the available text.\n"
        "- If the documents show that all Engineering BS programs use the ABET Student Outcomes (1)–(7), "
        "you may state that engineering programs have seven PLOs aligned with those outcomes.\n\n"
        "Remember: respond as APQA, do not mention that you are an AI, and do not refer to \"context\" "
        "in your replies.\n\n"
        "APQA reference information:\n"
        "{context}"
    )
    return system_prompt


# -------------------------------------------------------------------
# Build RAG chain
# -------------------------------------------------------------------

def build_rag_chain(config: RagConfig):
    if not retriever:
        logger.error("Retriever is not available; RAG chain cannot be built.")
        return None

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY missing; cannot build LLM.")
        return None

    # Explicitly pass the API key to ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2,
        max_output_tokens=4096,
        google_api_key=GOOGLE_API_KEY
    )

    system_prompt_text = get_system_prompt(config.language)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, question_answer_chain)

    def chain(inputs: Dict[str, Any]) -> str:
        response = rag.invoke({
            "input": inputs["input"],
            "chat_history": inputs.get("chat_history", []),
        })
        return response["answer"]

    logger.info(f"RAG chain built for language = {config.language}")
    return chain

rag_chain_en = build_rag_chain(RagConfig(language="en"))
rag_chain_ar = build_rag_chain(RagConfig(language="ar"))
rag_chain_fr = build_rag_chain(RagConfig(language="fr"))

# -------------------------------------------------------------------
# Flask app - API ONLY (no template rendering)
# -------------------------------------------------------------------

app = Flask(__name__)

# ✅ CORS Configuration - Allow requests from your frontend domain
CORS(app, origins=[
    "https://apqa.gastli.org",
    "http://apqa.gastli.org",
    "https://www.apqa.gastli.org",
    "http://www.apqa.gastli.org"
])

chat_history: List[str] = []

@app.route("/")
def index():
    """Simple API status endpoint"""
    return jsonify({
        "status": "ok",
        "message": "APQA Assessment Assistant API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        }
    })

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json(force=True) or {}
        message = (data.get("message") or "").strip()
        language = (data.get("language") or "en").lower()

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Choose the chain
        if language == "ar":
            chain = rag_chain_ar
        elif language == "fr":
            chain = rag_chain_fr
        else:
            chain = rag_chain_en

        if chain is None:
            logger.error("RAG chain is not available (check index or API key).")
            return jsonify({
                "reply": "System is not fully initialized (index or model issue). Please contact APQA."
            }), 503

        history_text = "\n".join(chat_history[-6:])
        logger.info(f"Processing message [{language}]: {message[:80]}...")

        answer = chain({
            "input": message,
            "chat_history": history_text,
        })

        chat_history.append(f"User: {message}")
        chat_history.append(f"Assistant: {answer}")

        return jsonify({"reply": answer})

    except Exception as e:
        logger.error(f"❌ Error in /chat: {e}", exc_info=True)
        return jsonify({
            "reply": "Sorry, something went wrong. Please try again.",
            "error": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "has_index": vector_index is not None,
        "has_api_key": bool(GOOGLE_API_KEY),
        "chains_ready": {
            "en": rag_chain_en is not None,
            "ar": rag_chain_ar is not None,
            "fr": rag_chain_fr is not None
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
