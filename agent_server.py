import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
# Prefer setting GEMINI_API_KEY in the environment (Render / local)
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- GLOBAL VARIABLES ---
rag_chain = None


def setup_agent() -> None:
    """Load FAISS index and build the RAG chain."""
    global rag_chain
    print("--- Loading Vector Index ---")

    try:
        # 1. Load embeddings and FAISS index
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

        # Retriver: show the model enough of the manuals to see full PLO lists
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 12,  # tune between 10–15 as needed
            }
        )
        print("✅ Vector Index Loaded")

        # 2. Setup Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.1,        # almost deterministic, but a bit flexible
            max_output_tokens=1200, # allow detailed answers
        )

        # 3. System prompt tuned for detailed, precise APQA answers
        system_prompt = (
            "You are an expert APQA Assessment Assistant for Qatar University. "
            "Your job is to answer questions using ONLY the provided context from APQA "
            "manuals, rubrics, and guidelines.\n\n"
            "When you answer:\n"
            "- Be clear, structured, and reasonably detailed (2–4 short paragraphs or a bullet list).\n"
            "- When the user asks about PEOs, PLOs, or Performance Indicators (PIs), "
            "first scan ALL provided context for any numbered lists or bullet lists that define them. "
            "Then:\n"
            "  * Count how many items are in the list (e.g., 7 outcomes) and say the number explicitly.\n"
            "  * List each outcome/indicator with its full wording as far as it appears in the context.\n"
            "- If the context shows the ABET Student Outcomes (1)–(7) that apply to all engineering BS programs, "
            "you may state that engineering programs have 7 PLOs aligned with those outcomes.\n"
            "- If a list looks truncated in the context (only items (1)–(3) visible), say clearly that "
            "only part of the list is visible and you cannot see the remaining items.\n"
            "- Base your answer strictly on the context you see, but simple reasoning like counting items in a list, "
            "or recognizing that a list labeled '(1)–(7)' means there are 7 outcomes, is allowed.\n"
            "- Only if the relevant information truly does NOT appear in the context at all, say exactly: "
            "\"I couldn't find that in the documents.\" Do NOT invent or guess.\n"
            "- Do NOT use citation tags like [1] or (Source: ...).\n\n"
            "Context:\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # 4. Build the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print("--- Agent Ready ---")

    except Exception as e:
        print(f"CRITICAL ERROR IN setup_agent: {e}")


# Initialize the agent at startup
setup_agent()


@app.route("/")
def home():
    return "APQA Agent is Live!"


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json or {}
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message"}), 400

    if not rag_chain:
        # Index or chain failed to initialize
        return jsonify({"error": "Agent is starting..."}), 503

    try:
        response = rag_chain.invoke({"input": user_message})
        # The retrieval chain returns a dict with an 'answer' key
        return jsonify({"reply": response.get("answer", "")})
    except Exception as e:
        print(f"Error while handling /chat: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Local debugging only; Render will use gunicorn
    app.run(host="0.0.0.0", port=10000)
