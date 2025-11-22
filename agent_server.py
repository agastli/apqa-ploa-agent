import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ✅ SAFER IMPORTS (Points to specific files)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

# ... (Rest of the code remains exactly the same)


# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- GLOBAL VARIABLES ---
rag_chain = None

def setup_agent():
    global rag_chain
    print("--- Loading Vector Index ---")
    
    try:
        # 1. Load Index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        if not os.path.exists("faiss_index"):
             print("⚠️ Error: 'faiss_index' folder not found!")
             return

        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("✅ Vector Index Loaded")

        # 2. Setup Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

        # 3. Create Prompt
        system_prompt = (
            "You are an expert APQA Assistant for Qatar University. "
            "Use the following context to answer the question. "
            "If the answer is not in the context, say 'I couldn't find that in the documents.' "
            "Do NOT use citation tags."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 4. Create Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("--- Agent Ready ---")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

setup_agent()

@app.route('/')
def home():
    return "APQA Agent is Live!"

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    user_message = data.get('message')
    if not user_message: return jsonify({"error": "No message"}), 400
    
    if not rag_chain:
        return jsonify({"error": "Agent is starting..."}), 503

    try:
        response = rag_chain.invoke({"input": user_message})
        return jsonify({"reply": response['answer']})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)