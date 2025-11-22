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

        # Show the model enough of the manuals to see full PLO lists, etc.
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 12,  # tune between 10–15 as needed
            }
        )
        print("✅ Vector Index Loaded")

        # 2. Setup Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.2,        # a bit more natural, still factual
            max_output_tokens=1200,
        )

        # 3. System prompt tuned for APQA-style answers (no “context says”)
        system_prompt = (
            "You represent Qatar University's Office of Academic Planning & Quality Assurance (APQA). "
            "Answer questions the way a knowledgeable APQA staff member would respond to faculty or programs. "
            "Use clear, professional, and friendly language. Do not sound robotic.\n\n"
            "STYLE AND TONE\n"
            "- Speak directly, as if you are part of APQA (e.g., \"Currently, the OAS supports...\"), "
            "not as an AI system.\n"
            "- Do NOT mention words like \"context\", \"documents\", \"manuals\", \"PDF\", or \"retrieved text\" "
            "in your answer.\n"
            "- Avoid phrases such as \"Based on the provided context\" or \"The context states\". "
            "Instead, state the information directly, as guidance from APQA.\n"
            "- When helpful, you may use short bullet points, but keep the explanation concise and human-sounding.\n\n"
            "POLICY AND GROUNDEDNESS\n"
            "- Your knowledge comes ONLY from the internal APQA manuals, rubrics, and guidelines provided below. "
            "You must not invent new policies or procedures.\n"
            "- Normal reasoning is allowed (for example, counting how many outcomes appear in a numbered list), "
            "but you must not fabricate requirements that are not supported by the information you see.\n"
            "- If something is clearly described, answer confidently in APQA's voice.\n"
            "- If something is NOT described, be honest and say something like: "
            "\"Our current APQA documents do not specify this point\" or "
            "\"This is not explicitly detailed in the available guidelines.\" "
            "Then, if appropriate, you may offer a neutral, practical suggestion.\n\n"
            "SPECIAL HANDLING FOR PEOs / PLOs / PERFORMANCE INDICATORS\n"
            "- When the user asks about PEOs, PLOs, Student Outcomes (SOs), or Performance Indicators (PIs):\n"
            "  * First, scan all the information you see for numbered lists or bullet lists that define them.\n"
            "  * Count how many items are in the list and state the number explicitly (e.g., \"Engineering BS programs "
            "have seven outcomes\").\n"
            "  * List each outcome/indicator with its wording, as fully as it appears in the information you see.\n"
            "  * If the list appears incomplete (for example, only items (1)–(3) are visible), say clearly that the "
            "remaining items are not shown in the available text.\n"
            "- If the documents show that all Engineering BS programs use the ABET Student Outcomes (1)–(7), you may "
            "state that engineering programs have seven PLOs aligned with those outcomes.\n\n"
            "Remember: respond as APQA, do not mention that you are an AI, and do not refer to \"context\" in your replies.\n\n"
            "APQA reference information:\n"
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
