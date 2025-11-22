import os
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


# -------------------------------------------------------------------
# Configuration & basic checks
# -------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Please set GEMINI_API_KEY in your environment before running agent_server.py"
    )

INDEX_DIR = os.environ.get("INDEX_DIR", "faiss_index")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


@dataclass
class RagConfig:
    language: str = "en"  # "en", "ar", or "fr"


# -------------------------------------------------------------------
# Load FAISS index
# -------------------------------------------------------------------

def load_vector_index() -> FAISS:
    """Load the FAISS index built by build_index.py."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    if not os.path.isdir(INDEX_DIR):
        raise RuntimeError(
            f"Index directory '{INDEX_DIR}' not found. "
            "Run build_index.py locally first to create the FAISS index."
        )

    index = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        index_name="index",
        allow_dangerous_deserialization=True,
    )
    return index


vector_index = load_vector_index()
retriever = vector_index.as_retriever(search_kwargs={"k": 6})


# -------------------------------------------------------------------
# Helper: system prompts per language
# -------------------------------------------------------------------

def get_system_prompt(language: str) -> str:
    """Return a friendly APQA-style system prompt in the requested language."""
    if language == "ar":
        return (
            "أنت مساعد مكتب التخطيط والجودة الأكاديمية بجامعة قطر.\n"
            "دورك هو مساعدة الزملاء في برامج الكليات على فهم وتطبيق:\n"
            "• مخرجات التعلم البرنامجية PLOs\n"
            "• أدوات التقييم (مثل rubrics)\n"
            "• نظام التقييم الإلكتروني OAS\n\n"
            "اعتمد أولاً على الوثائق المتاحة في سياقك (الملفات التي تم تغذيتك بها).\n"
            "إن لم تجد إجابة واضحة في الوثائق، وضّح ذلك بصراحة، "
            "وقدّم شرحاً عاماً مبسطاً استناداً إلى أفضل الممارسات في الجودة والاعتماد.\n\n"
            "اكتب بأسلوب مهني ولطيف كأنك موظف من مكتب APQA يجيب عن استفسار زميل.\n"
            "إذا احتاج الأمر إلى تنبيه أو توضيح رسمي، استخدم عبارات مثل:\n"
            "«في مكتب التخطيط والجودة الأكاديمية يتم عادةً…» أو «حسب الممارسات المعتمدة في الجامعة…».\n\n"
            "السياق التالي يحتوي على المقتطفات ذات الصلة من الوثائق:\n"
            "{context}\n\n"
            "تاريخ المحادثة السابقة بين المستخدم والمساعد:\n"
            "{chat_history}\n\n"
            "سؤال الزميل:\n"
            "{input}\n"
            "قدّم إجابة واضحة ومنظمة، مع نقاط مرقمة إن لزم الأمر."
        )

    elif language == "fr":
        return (
            "Vous êtes l’assistant du Bureau de la planification et de l’assurance qualité académique "
            "de l’Université du Qatar.\n"
            "Votre rôle est d’aider les collègues à comprendre et appliquer :\n"
            "• les PLO (Program Learning Outcomes)\n"
            "• les outils d’évaluation (rubrics, etc.)\n"
            "• le système d’évaluation en ligne (OAS).\n\n"
            "Appuyez-vous d’abord sur les documents fournis dans le contexte.\n"
            "Si l’information n’est pas disponible, dites-le clairement et proposez une explication générale "
            "basée sur les bonnes pratiques en assurance qualité.\n\n"
            "Répondez dans un style professionnel mais convivial, comme un collègue du bureau APQA.\n\n"
            "Contexte documentaire :\n"
            "{context}\n\n"
            "Historique de la conversation :\n"
            "{chat_history}\n\n"
            "Question de l’utilisateur :\n"
            "{input}\n"
            "Donnez une réponse structurée et précise."
        )

    # Default: English
    return (
        "You are the assistant of the Office of Academic Planning & Quality Assurance (APQA) at Qatar University.\n"
        "Your role is to help colleagues understand and apply:\n"
        "• Program Learning Outcomes (PLOs)\n"
        "• Assessment tools (rubrics, performance indicators)\n"
        "• The Online Assessment System (OAS) and related procedures.\n\n"
        "Always rely first on the official assessment documents provided in the context. "
        "If the documents do not contain a direct answer, say this explicitly and then provide a general explanation "
        "based on good practices in academic quality and accreditation.\n\n"
        "Write in a friendly, collegial tone as if you are an APQA staff member answering a colleague’s question. "
        "Avoid robotic phrasing such as ‘the context states’. Instead, use phrases like "
        "‘In APQA, we usually…’ or ‘According to the current assessment guidelines…’ when appropriate.\n\n"
        "Relevant document excerpts:\n"
        "{context}\n\n"
        "Conversation so far:\n"
        "{chat_history}\n\n"
        "User question:\n"
        "{input}\n"
        "Provide a clear, organized answer. Use bullet points where helpful."
    )


# -------------------------------------------------------------------
# Build RAG chain using LangChain 1.x runnables (no langchain.chains)
# -------------------------------------------------------------------

def build_rag_chain(config: RagConfig):
    """Create a RAG chain for a given language."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2,
        max_output_tokens=1024,
    )

    # Prompt template uses {context}, {chat_history}, {input}
    system_prompt = get_system_prompt(config.language)
    prompt = ChatPromptTemplate.from_template(system_prompt)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Runnable pipeline:
    # 1. Use retriever to get docs and format them as context
    # 2. Pass context + input (+ chat_history) to the prompt
    # 3. Generate with LLM
    # 4. Parse to plain string
    rag_runnable = (
        RunnableParallel(
            {
                "context": retriever | (lambda docs: format_docs(docs)),
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap into a callable that matches our older interface
    def chain(inputs: Dict[str, Any]) -> str:
        return rag_runnable.invoke(
            {
                "input": inputs["input"],
                "chat_history": inputs.get("chat_history", ""),
            }
        )

    return chain


# Create one chain per language
rag_chain_en = build_rag_chain(RagConfig(language="en"))
rag_chain_ar = build_rag_chain(RagConfig(language="ar"))
rag_chain_fr = build_rag_chain(RagConfig(language="fr"))


# -------------------------------------------------------------------
# Flask app & in-memory chat history
# -------------------------------------------------------------------

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Very simple global history (sufficient for your single-user deployment)
chat_history: List[str] = []


@app.route("/")
def index() -> Any:
    """Serve the main HTML page."""
    return app.send_static_file("index.html")


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    POST /chat
    body: { "message": "...", "language": "en" | "ar" | "fr" }
    """
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    language = (data.get("language") or "en").lower()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # Choose chain based on language
    if language == "ar":
        chain = rag_chain_ar
    elif language == "fr":
        chain = rag_chain_fr
    else:
        chain = rag_chain_en

    # Build a simple textual chat history (last 10 turns)
    history_text = "\n".join(chat_history[-10:])

    try:
        answer = chain(
            {
                "input": message,
                "chat_history": history_text,
            }
        )
    except Exception as e:
        # In production you may want to log this error
        answer = (
            "Sorry, something went wrong while generating the answer. "
            "Please try again or contact APQA for assistance."
        )

    # Update in-memory history
    chat_history.append(f"User: {message}")
    chat_history.append(f"Assistant: {answer}")

    return jsonify({"reply": answer})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
