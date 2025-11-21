import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# ✅ FIX: Import from langchain_core
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found in Environment!")
else:
    os.environ["GOOGLE_API_KEY"] = API_KEY

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- GLOBAL VARIABLES ---
qa_chain = None

def setup_agent():
    global qa_chain
    print("--- Loading Vector Index ---")
    
    try:
        # 1. Load the pre-built index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Load from the uploaded folder
        if not os.path.exists("faiss_index"):
             print("⚠️ Error: 'faiss_index' folder not found on server!")
             return

        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector Index Loaded Successfully")

        # 2. Setup Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # 3. Create Prompt
        template = """
        You are an expert APQA Assistant for Qatar University.
        Use the following context to answer the question.
        
        GUIDELINES:
        - Provide a clear, synthesized answer.
        - Use bold headers and bullet points.
        - If the answer is not in the context, say "I couldn't find that in the documents."
        
        Context: {context}

        Question: {question}
        
        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # 4. Create Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        print("--- Agent Ready ---")

    except Exception as e:
        print(f"CRITICAL ERROR loading index: {e}")

# Initialize
setup_agent()

@app.route('/')
def home():
    return "APQA Agent (RAG Version) is Live!"

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    user_message = data.get('message')
    if not user_message: return jsonify({"error": "No message"}), 400
    
    if not qa_chain:
        return jsonify({"error": "Agent is starting or failed to load index."}), 503

    try:
        response = qa_chain.invoke(user_message)
        return jsonify({"reply": response['result']})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)