import os
import uuid

from flask import Flask, request, jsonify, session
from flask_cors import CORS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

app = Flask(__name__)
# Needed for Flask session cookies
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "apqa-secret-key")
CORS(app, resources={r"/*": {"origins": "*"}})

# --- GLOBAL VARIABLES ---
rag_chain = None
conversational_rag = None

# Store chat history per session_id in server memory
history_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) the chat history for a given session."""
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]


def setup_agent() -> None:
    """Load FAISS index and build the RAG chain with conversation memory."""
    global rag_chain, conversational_rag
    print("--- Loading Vector Index ---")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        if not os.path.exists("faiss_index"):
            print("⚠️ Error: 'faiss_index' folder not found!")
            return

        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 12}
        )
        print("✅ Vector Index Loaded")

        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.2,
            max_output_tokens=1200,
        )

        # --- System prompt with history support ---
        system_prompt = (
            "You represent Qatar University's Office of Academic Planning & Quality Assurance (APQA). "
            "Answer questions the way a knowledgeable APQA staff member would respond to faculty or programs. "
            "Use clear, professional, and friendly language. Do not sound robotic.\n\n"
            "STYLE AND TONE\n"
            "- Speak directly, as if you are part of APQA (e.g., \"Currently, the OAS supports...\").\n"
            "- Do NOT mention words like \"context\", \"documents\", \"manuals\", or \"PDF\" in your answer.\n"
            "- Avoid phrases such as \"Based on the provided context\" or \"The context states\". "
            "Instead, state the information directly, as guidance from APQA.\n"
            "- When helpful, you may use short bullet points, but keep the explanation concise and human-sounding.\n\n"
            "POLICY AND GROUNDEDNESS\n"
            "- Your knowledge comes ONLY from the internal APQA manuals, rubrics, and guidelines provided below. "
            "Do not invent new policies.\n"
            "- Normal reasoning is allowed (e.g., counting the number of outcomes in a list).\n"
            "- If something is NOT described, say: "
            "\"Our current APQA documents do not specify this point\" or "
            "\"This is not explicitly detailed in the available guidelines.\" "
            "You may then offer a neutral, practical suggestion.\n\n"
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

        # IMPORTANT: include {history} so previous turns are visible to the model
        from langchain_core.prompts import MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )


        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        # Wrap with conversation memory
        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

        print("--- Agent Ready ---")

    except Exception as e:
        print(f"CRITICAL ERROR IN setup_agent: {e}")


# Initialize at startup
setup_agent()


@app.route("/")
def home():
    return "APQA Agent is Live!"


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    global conversational_rag
    data = request.json or {}
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message"}), 400

    if conversational_rag is None:
        return jsonify({"error": "Agent is starting..."}), 503

    # Ensure each browser gets a stable session_id via Flask cookies
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]

    try:
        response = conversational_rag.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}},
        )
        return jsonify({"reply": response.get("answer", "")})
    except Exception as e:
        print(f"Error while handling /chat: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
