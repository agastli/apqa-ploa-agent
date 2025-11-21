import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

# --- CONFIGURATION ---
API_KEY = "AIzaSyCms2tGXckFItuCtmNZTp8wshZJyAmN2b8"
KNOWLEDGE_FOLDER = "knowledge"

app = Flask(__name__)
CORS(app)

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
    
    # 1. SMART UPLOAD: Check cloud first
    if os.path.exists(KNOWLEDGE_FOLDER):
        local_files = os.listdir(KNOWLEDGE_FOLDER)
        # Get list of what is already on Google
        cloud_files = get_existing_files() 
        
        print(f"Scanning {len(local_files)} local files...")
        
        for filename in local_files:
            if not filename.lower().endswith(".pdf"): continue
            
            path = os.path.join(KNOWLEDGE_FOLDER, filename)
            
            # --- THE SPEED FIX ---
            if filename in cloud_files:
                print(f"⚡ Found '{filename}' in cloud. Skipping upload.")
                uploaded_file = cloud_files[filename]
                
                # Quick check to ensure it's active
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

            # Add to knowledge base
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

    # 3. Create Session (Using the fast 1.5-flash model)
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
    print("--- Agent Ready on Port 5000 ---")

# Initialize agent on startup
setup_agent()

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
    app.run(port=5000, debug=False)