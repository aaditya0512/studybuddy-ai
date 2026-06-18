import os
from flask import Flask, render_template, redirect, url_for, flash, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from config import Config
from models import db, User, Document, ChatHistory
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

# Initialize DB and LoginManager
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables and directories
with app.app_context():
    db.create_all()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    if current_user.is_authenticated:
        recent_docs = Document.query.filter_by(user_id=current_user.id).order_by(Document.upload_date.desc()).limit(5).all()
        return render_template('index.html', recent_docs=recent_docs)
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('register'))
            
        new_user = User(name=name, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
            
        login_user(user)
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    docs = Document.query.filter_by(user_id=current_user.id).all()
    doc_count = len(docs)
    chat_count = ChatHistory.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', docs=docs, doc_count=doc_count, chat_count=chat_count)

from flask import jsonify
from modules.pdf_loader import extract_text_from_pdf
from modules.chunker import chunk_text
from modules.retriever import store_chunks
from modules.chatbot import generate_answer
from modules.summarizer import generate_summary
from modules.quiz_generator import generate_quiz, generate_flashcards

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'document' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['document']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the document
        text = extract_text_from_pdf(filepath)
        
        # Ephemeral file cleanup
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing ephemeral file: {e}")
            
        if text:
            chunks = chunk_text(text)
            user_collection = f"user_{current_user.id}"
            
            try:
                store_chunks(chunks, user_collection)
            except Exception as e:
                flash(f'Vector store error: {str(e)}')
                return redirect(url_for('index'))
                
            new_doc = Document(user_id=current_user.id, file_name=filename, text_content=text)
            db.session.add(new_doc)
            db.session.commit()
            flash('Document uploaded and processed successfully!')
        else:
            flash('Error extracting text from PDF')
    else:
        flash('Only PDF files are supported currently.')
    return redirect(url_for('index'))

@app.route('/chat', methods=['GET'])
@login_required
def chat_page():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.json
    query = data.get('question')
    if not query:
        return jsonify({'error': 'No question provided'}), 400
        
    user_collection = f"user_{current_user.id}"
    try:
        answer, context = generate_answer(query, collection_name=user_collection)
        
        # Save chat history
        chat_entry = ChatHistory(user_id=current_user.id, question=query, answer=answer)
        db.session.add(chat_entry)
        db.session.commit()
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summary')
@login_required
def summary_page():
    docs = Document.query.filter_by(user_id=current_user.id).all()
    return render_template('summary.html', docs=docs)

@app.route('/api/generate', methods=['POST'])
@login_required
def api_generate():
    data = request.json
    doc_id = data.get('doc_id')
    gen_type = data.get('type') # summary, notes, quiz, flashcards
    
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first()
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
        
    text = doc.text_content
    if not text:
        # Fallback to local file if text_content is empty (for backward compatibility before migration)
        if getattr(doc, 'filepath', None) and os.path.exists(doc.filepath):
            text = extract_text_from_pdf(doc.filepath)
        if not text:
            return jsonify({'error': 'Could not read document'}), 500
        
    try:
        if gen_type == 'summary':
            result = generate_summary(text, length="medium")
            return jsonify({'result': result})
        elif gen_type == 'notes':
            result = generate_summary(text, length="detailed")
            return jsonify({'result': result})
        elif gen_type == 'quiz':
            result = generate_quiz(text, num_questions=5)
            return jsonify({'result': result})
        elif gen_type == 'flashcards':
            result = generate_flashcards(text, num_cards=5)
            return jsonify({'result': result})
        else:
            return jsonify({'error': 'Invalid generation type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
