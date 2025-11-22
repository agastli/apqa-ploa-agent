import os
import uuid

from flask import Flask, request, jsonify, session
from flask_cors import CORS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "apqa-secret-key")
CORS(app, resources={r"/*": {"origins": "*"}})

history_store: dict[str, InMemoryChatMessageHistory] = {}

llm = None
conversational_rag = None


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]


# ---------------------------------------------------------------------
# AGENT SETUP
# ---------------------------------------------------------------------
def setup_agent() -> None:
    global llm, conversational_rag
    print("--- Loading Vector Index ---")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        if not os.path.exists("faiss_index"):
            print("⚠️ Error: 'faiss_index' folder not found")
            return

        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": 12}
        )
        print("✅ Vector Index Loaded")

        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.2,
            max_output_tokens=1200,
        )

        # 1) History-aware retriever: rewrite follow-up questions
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant that rewrites follow-up questions into "
                    "standalone questions using the chat history. "
                    "Do NOT answer the question. Only rewrite when needed.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=base_retriever,
            prompt=contextualize_prompt,
        )

        # 2) QA chain prompt with language and history
        system_prompt = (
            "You represent Qatar University's Office of Academic Planning & Quality Assurance (APQA). "
            "Answer questions the way a knowledgeable APQA staff member would respond to faculty or programs. "
            "Use clear, professional, and friendly language and avoid sounding robotic.\n\n"
            "LANGUAGE CONTROL\n"
            "- You receive a parameter called 'language'.\n"
            "- If language == 'ar', answer entirely in Modern Standard Arabic.\n"
            "- If language == 'en', answer entirely in English.\n"
            "- Do not mix languages unless explicitly requested.\n\n"
            "STYLE AND TONE\n"
            "- Speak directly, as if you are part of APQA (e.g., \"Currently, we use...\").\n"
            "- Do NOT mention words like \"context\", \"documents\", or \"PDF\".\n"
            "- Avoid phrases such as \"Based on the provided context\" or \"The context states\".\n"
            "- You may use short bullet points when helpful.\n\n"
            "POLICY & SOURCES\n"
            "- Use ONLY the APQA manuals and assessment documents provided below as your knowledge source.\n"
            "- Normal reasoning is allowed (e.g., counting outcomes in a list).\n"
            "- If something is not documented, say that it is not specified in the available APQA materials, "
            "then offer a neutral, practical suggestion.\n\n"
            "FOLLOW-UP REQUESTS (REWRITE / TRANSLATE)\n"
            "- If the user asks to rewrite, translate, summarise, or clarify your previous answer, "
            "carefully read the previous assistant messages in the chat_history and operate on your last answer. "
            "- In that case, you may ignore the APQA reference documents and just transform your previous answer.\n\n"
            "PLOS / OUTCOMES / PERFORMANCE INDICATORS\n"
            "- When the user asks about PEOs, PLOs, Student Outcomes (SOs), or Performance Indicators (PIs):\n"
            "  * State how many there are.\n"
            "  * List them with their wording as given in the documents.\n"
            "  * If the list appears incomplete, say that the rest do not appear in the available text.\n\n"
            "APQA reference information:\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "Language: {language}\n\n"
                    "User question: {input}"
                ),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain,
        )

        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        print("--- Agent Ready ---")

    except Exception as e:
        print(f"CRITICAL ERROR in setup_agent: {e}")


setup_agent()

# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return "APQA Agent is Live!"


def _looks_like_rewrite_request(text: str) -> bool:
    """Basic heuristic for 'rewrite / translate my last answer'."""
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "rewrite in arabic",
            "write in arabic",
            "translate to arabic",
            "re-write in arabic",
            "rewrite answer",
            "translate your answer",
            "اكتبها بالعربي",
            "أعد صياغتها بالعربية",
        ]
    )


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    global llm, conversational_rag
    data = request.json or {}
    user_message = data.get("message", "").strip()
    language = data.get("language", "en")

    if not user_message:
        return jsonify({"error": "No message"}), 400

    if conversational_rag is None or llm is None:
        return jsonify({"error": "Agent is starting..."}), 503

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]
    history = get_session_history(session_id)

    # Special handling: user wants to rewrite/translate the previous answer
    if _looks_like_rewrite_request(user_message) and history.messages:
        last_ai = None
        # find last assistant message
        for msg in reversed(history.messages):
            if getattr(msg, "type", "") == "ai":
                last_ai = msg.content
                break

        if last_ai is None:
            # Fall back to normal RAG if no previous answer
            pass
        else:
            target_lang = "Modern Standard Arabic" if language == "ar" else "English"
            prompt = (
                "You are rewriting your previous APQA answer for a user.\n"
                f"Target language: {target_lang}.\n"
                "Rewrite the following answer in the target language, preserving the meaning and structure, "
                "and keeping the same APQA tone. Do not add new information and do not mention that you are "
                "rewriting or translating.\n\n"
                f"{last_ai}"
            )

            try:
                result = llm.invoke(prompt)
                # also push this back into history so it is part of the conversation
                history.add_ai_message(result.content)
                return jsonify({"reply": result.content})
            except Exception as e:
                print(f"Error in translate/rewrite path: {e}")
                return jsonify({"error": str(e)}), 500

    # Normal RAG conversational path
    try:
        response = conversational_rag.invoke(
            {"input": user_message, "language": language},
            config={"configurable": {"session_id": session_id}},
        )
        return jsonify({"reply": response.get("answer", "")})
    except Exception as e:
        print(f"Error while handling /chat: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local testing only; Render will run via gunicorn
    app.run(host="0.0.0.0", port=10000)
