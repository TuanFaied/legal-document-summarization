# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import PyPDF2
import docx
import io
import re

app = Flask(__name__)
CORS(app)

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "AventIQ-AI/t5-summarization-for-legal-contracts"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Predefined clause categories (can be extended)
CLAUSE_CATEGORIES = {
    "indemnification": ["indemnify", "indemnification", "hold harmless"],
    "confidentiality": ["confidential", "non-disclosure", "nda"],
    "termination": ["termination", "terminate", "expiration"],
    "governing law": ["governing law", "jurisdiction", "venue"],
    "limitation of liability": ["limitation of liability", "cap on damages", "consequential damages"],
    "force majeure": ["force majeure", "act of god", "unforeseeable circumstances"],
    "intellectual property": ["intellectual property", "ip", "patent", "copyright", "trademark"]
}

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file format")

def split_into_clauses(text):
    # Simple heuristic to split contract into clauses
    # This can be improved with more sophisticated NLP techniques
    clauses = re.split(r'\n\s*\d+[\.\)]\s*|\n\s*ARTICLE\s+\d+|\n\s*SECTION\s+\d+', text)
    return [clause.strip() for clause in clauses if clause.strip()]

def classify_clause(clause_text):
    clause_text_lower = clause_text.lower()
    scores = {}
    
    for category, keywords in CLAUSE_CATEGORIES.items():
        score = sum(1 for keyword in keywords if keyword in clause_text_lower)
        if score > 0:
            scores[category] = score
    
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    return "unknown"

def summarize_clause(clause_text):
    input_text = "summarize: " + clause_text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        length_penalty=0.8,
        early_stopping=True
    )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Extract text from the uploaded file
        text = extract_text_from_file(file)
        
        # Split into clauses
        clauses = split_into_clauses(text)
        
        # Process each clause
        results = []
        for i, clause in enumerate(clauses):
            if len(clause) < 50:  # Skip very short clauses
                continue
                
            category = classify_clause(clause)
            summary = summarize_clause(clause)
            
            results.append({
                "clause_id": i+1,
                "text": clause[:200] + "..." if len(clause) > 200 else clause,  # Preview
                "category": category,
                "summary": summary
            })
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        summary = summarize_clause(data['text'])
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)