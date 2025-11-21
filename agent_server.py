import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# 1. Try to get key from Render Environment, otherwise use your hardcoded key
# (This allows it to work on Render securely AND on your laptop)
# ✅ SECURE WAY: Only look for the key in the environment variables.
# Do NOT paste the actual key starting with "AIza..." here.
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("⚠️ Error: GEMINI_API_KEY not found in environment variables!")
KNOWLEDGE_FOLDER = "knowledge"

app = Flask(__name__)

# --- THE CONNECTION FIX (CORS) ---
# This specific configuration allows Hostinger (and everyone else) to talk to Render
CORS(app, resources={r"/*": {"origins": "*"}})

# --- GLOBAL VARIABLES ---
client = genai.Client(api_key=API_KEY)
chat_session = None

def get_existing_files():
    """Creates a dictionary of files already uploaded to Google."""
    print("Checking for existing files on Google Cloud...")
    existing = {}
    try:
        # List files currently stored in your project
        for f in client.files.list():
            if f.display_name:
                existing[f.display_name] = f
    except Exception as e:
        print(f"Warning: Could not list existing files: {e}")
    return existing

def setup_agent():
    global chat_session
    print("--- Initializing Agent ---")
    
    knowledge_base_parts = []
    
    # 1. SMART UPLOAD: Check cloud first (KEEPING THIS FOR SPEED)
    if os.path.exists(KNOWLEDGE_FOLDER):
        local_files = os.listdir(KNOWLEDGE_FOLDER)
        cloud_files = get_existing_files() 
        
        print(f"Scanning {len(local_files)} local files...")
        
        for filename in local_files:
            if not filename.lower().endswith(".pdf"): continue
            
            path = os.path.join(KNOWLEDGE_FOLDER, filename)
            
            # --- THE SPEED FIX ---
            if filename in cloud_files:
                print(f"⚡ Found '{filename}' in cloud. Skipping upload.")
                uploaded_file = cloud_files[filename]
                
                if uploaded_file.state.name != "ACTIVE":
                    print(f"   (File is {uploaded_file.state.name}, waiting...)")
                    while uploaded_file.state.name == "PROCESSING":
                        time.sleep(1)
                        uploaded_file = client.files.get(name=uploaded_file.name)
            else:
                # Only upload if it's NOT in the cloud
                try:
                    print(f"⬆️  Uploading '{filename}'...", end=" ")
                    uploaded_file = client.files.upload(file=path, config={'display_name': filename})
                    
                    while uploaded_file.state.name == "PROCESSING":
                        time.sleep(1)
                        uploaded_file = client.files.get(name=uploaded_file.name)
                    
                    if uploaded_file.state.name == "ACTIVE":
                        print("✅ Done.")
                    else:
                        print("❌ Failed.")
                        continue
                except Exception as e:
                    print(f"Error uploading {filename}: {e}")
                    continue

            knowledge_base_parts.append(
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type
                )
            )

    # 2. Configure Chat
    print("Configuring chat session...")
    tools = [types.Tool(google_search=types.GoogleSearch())]
    
    knowledge_base_parts.append(
        types.Part.from_text(text="Use these uploaded documents to answer questions.")
    )

    # 3. Create Session
    chat_session = client.chats.create(
        model="gemini-flash-latest", 
        
        config=types.GenerateContentConfig(
            system_instruction=[
                types.Part.from_text(text="""
                You are an expert APQA Assistant for Qatar University.
                YOUR GOAL: Provide clear, synthesized answers.
                STRICT GUIDELINES:
                1. SYNTHESIZE: Combine info into a summary.
                2. NO CITATION TAGS: Do NOT use.
                3. FORMATTING: Use bold headers and bullet points.
                4. SCOPE: Use files first, then Google Search.
                """),
            ],
            tools=tools,
            response_mime_type="text/plain",
        ),
        history=[
            types.Content(role="user", parts=knowledge_base_parts),
            types.Content(role="model", parts=[types.Part.from_text(text="Ready.")])
        ]
    )
    print("--- Agent Ready ---")

# Initialize on startup (with error handling)
try:
    setup_agent()
except Exception as e:
    print(f"Failed to init agent: {e}")

@app.route('/')
def home():
    return "APQA Agent is Live!"

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = chat_session.send_message(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render requires listening on 0.0.0.0
    app.run(host='0.0.0.0', port=10000)