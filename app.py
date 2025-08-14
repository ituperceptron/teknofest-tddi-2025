from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import sqlite3
import os
import sys
import logging
from datetime import datetime
import uuid
import shutil
import time

from models.model_manager import ModelManager
from models.classifier import classify_text
from models.summarizer import summarize_text
from models.ocr_processor import ocr_processor
from models.ner_processor import ner_processor

# Suppress verbose outputs and warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Redirect llama-cpp verbose output to devnull during model loading
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Global model manager instance (singleton)
model_manager = ModelManager()



def get_turkish_time():
    """Get current time in Turkish timezone (UTC+3)"""
    from datetime import timezone, timedelta
    turkish_tz = timezone(timedelta(hours=3))
    return datetime.now(turkish_tz)


def clear_memory_after_ocr():
    """Clear memory after Obje ve Karakter Tanıma operations to reduce RAM usage"""
    try:
        # Clear Obje ve Karakter Tanıma processor memory
        if hasattr(ocr_processor, '_clear_memory'):
            ocr_processor._clear_memory()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
            
    except Exception as e:
        print(f"Memory cleanup error: {e}")


app = Flask(__name__)
app.secret_key = os.urandom(24)  # More secure random secret key
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            content TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create PDF files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            file_id TEXT,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Add file_id column if it doesn't exist
    cursor.execute("PRAGMA table_info(pdf_files)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'file_id' not in columns:
        cursor.execute("ALTER TABLE pdf_files ADD COLUMN file_id TEXT")
    
    # Create classification PDF files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            file_id TEXT,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Add file_id column if it doesn't exist
    cursor.execute("PRAGMA table_info(classification_pdfs)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'file_id' not in columns:
        cursor.execute("ALTER TABLE classification_pdfs ADD COLUMN file_id TEXT")
    
    # Create Obje ve Karakter Tanıma PDF files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ocr_pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create Obje ve Karakter Tanıma image files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ocr_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create Varlık Tanıma PDF files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ner_pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Create text input tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summary_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ner_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create previous_works table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS previous_works (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            description TEXT,
            work_type TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create summary results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summary_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            summary_content TEXT,
            parent_input_id INTEGER,
            parent_input_type TEXT,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create classification results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            main_category TEXT,
            confidence_score TEXT,
            parent_input_id INTEGER,
            parent_input_type TEXT,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create Obje ve Karakter Tanıma results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ocr_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            ocr_text TEXT,
            source_type TEXT,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create Varlık Tanıma results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ner_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            ner_text TEXT,
            entities_json TEXT,
            source_type TEXT,
            upload_date TIMESTAMP DEFAULT (datetime('now', '+3 hours')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    
    conn.commit()
    conn.close()

# Create upload directories
def create_upload_dirs():
    upload_dir = 'uploads'
    summary_pdf_dir = os.path.join(upload_dir, 'summary_pdfs')
    classification_pdf_dir = os.path.join(upload_dir, 'classification_pdfs')
    ocr_pdf_dir = os.path.join(upload_dir, 'ocr_pdfs')  # Obje ve Karakter Tanıma PDF'leri
    ocr_image_dir = os.path.join(upload_dir, 'ocr_images')  # Obje ve Karakter Tanıma resimleri
    ner_pdf_dir = os.path.join(upload_dir, 'ner_pdfs')  # Varlık Tanıma PDF'leri
    summary_text_dir = os.path.join(upload_dir, 'summary_texts')
    classification_text_dir = os.path.join(upload_dir, 'classification_texts')
    ner_text_dir = os.path.join(upload_dir, 'ner_texts')  # Varlık Tanıma metinleri
    summary_results_dir = os.path.join(upload_dir, 'summary_results')
    classification_results_dir = os.path.join(upload_dir, 'classification_results')
    ocr_results_dir = os.path.join(upload_dir, 'ocr_results')  # Obje ve Karakter Tanıma sonuçları
    ner_results_dir = os.path.join(upload_dir, 'ner_results')  # Varlık Tanıma sonuçları
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    if not os.path.exists(summary_pdf_dir):
        os.makedirs(summary_pdf_dir)
    if not os.path.exists(classification_pdf_dir):
        os.makedirs(classification_pdf_dir)
    if not os.path.exists(ocr_pdf_dir):
        os.makedirs(ocr_pdf_dir)
    if not os.path.exists(ocr_image_dir):
        os.makedirs(ocr_image_dir)
    if not os.path.exists(ner_pdf_dir):
        os.makedirs(ner_pdf_dir)
    if not os.path.exists(summary_text_dir):
        os.makedirs(summary_text_dir)
    if not os.path.exists(classification_text_dir):
        os.makedirs(classification_text_dir)
    if not os.path.exists(ner_text_dir):
        os.makedirs(ner_text_dir)
    if not os.path.exists(summary_results_dir):
        os.makedirs(summary_results_dir)
    if not os.path.exists(classification_results_dir):
        os.makedirs(classification_results_dir)
    if not os.path.exists(ocr_results_dir):
        os.makedirs(ocr_results_dir)
    if not os.path.exists(ner_results_dir):
        os.makedirs(ner_results_dir)

# Initialize database and upload directories on startup
init_db()
create_upload_dirs()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash('Geçersiz kullanıcı adı veya şifre', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', (username, password, email))
            conn.commit()
            print("sorun yok")
            flash('Kayıt başarılı! Lütfen giriş yapın.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Kullanıcı adı zaten mevcut', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get summary PDFs (summarization works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'summary' as work_type, upload_date as created_at, 
               'Özetleme işlemi' as description, file_path
        FROM pdf_files 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    summary_works = cursor.fetchall()
    
    # Get classification PDFs (classification works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'classification' as work_type, upload_date as created_at,
               'Sınıflandırma işlemi' as description, file_path
        FROM classification_pdfs 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    classification_works = cursor.fetchall()
    
     # Get Obje ve Karakter Tanıma PDFs (Obje ve Karakter Tanıma works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ocr_pdf' as work_type, upload_date as created_at,
               'Obje ve Karakter Tanıma işlemi (PDF)' as description, file_path
        FROM ocr_pdfs 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ocr_pdf_works = cursor.fetchall()
    
    # Get Obje ve Karakter Tanıma images (Obje ve Karakter Tanıma works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ocr_image' as work_type, upload_date as created_at,
               'Obje ve Karakter Tanıma işlemi (Resim)' as description, file_path
        FROM ocr_images 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ocr_image_works = cursor.fetchall()
    
    # Get NER PDFs (NER works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ner' as work_type, upload_date as created_at,
               'NER işlemi' as description, file_path
        FROM ner_pdfs 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ner_works = cursor.fetchall()
    
    # Get text input works
    cursor.execute('''
        SELECT id, user_id, original_filename, 'summary_text' as work_type, upload_date as created_at, 
               'Metin Özetleme' as description, file_path
        FROM summary_texts 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    summary_text_works = cursor.fetchall()
    
    cursor.execute('''
        SELECT id, user_id, original_filename, 'classification_text' as work_type, upload_date as created_at,
               'Metin Sınıflandırma' as description, file_path
        FROM classification_texts 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    classification_text_works = cursor.fetchall()
    
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ner_text' as work_type, upload_date as created_at,
               'Metin Varlık Tanıma İşlemi' as description, file_path
        FROM ner_texts 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ner_text_works = cursor.fetchall()
    
    # Get summary results
    cursor.execute('''
        SELECT sr.id, sr.user_id, sr.original_filename, 'summary_result' as work_type, sr.upload_date as created_at,
               'Özet Sonucu' as description, sr.file_path
        FROM summary_results sr
        WHERE sr.user_id = ? 
        ORDER BY sr.upload_date DESC
    ''', (session['user_id'],))
    summary_result_works = cursor.fetchall()
    
    # Get classification results
    cursor.execute('''
        SELECT cr.id, cr.user_id, cr.original_filename, 'classification_result' as work_type, cr.upload_date as created_at,
               'Sınıflandırma Sonucu' as description, cr.file_path
        FROM classification_results cr
        WHERE cr.user_id = ? 
        ORDER BY cr.upload_date DESC
    ''', (session['user_id'],))
    classification_result_works = cursor.fetchall()
    
    # Get Obje ve Karakter Tanıma results
    cursor.execute('''
        SELECT or_table.id, or_table.user_id, or_table.original_filename, 'ocr_result' as work_type, or_table.upload_date as created_at,
               'Obje ve Karakter Tanıma Sonucu' as description, or_table.file_path
        FROM ocr_results or_table
        WHERE or_table.user_id = ? 
        ORDER BY or_table.upload_date DESC
    ''', (session['user_id'],))
    ocr_result_works = cursor.fetchall()
    
    # Get Varlık Tanıma results
    cursor.execute('''
        SELECT nr.id, nr.user_id, nr.original_filename, 'ner_result' as work_type, nr.upload_date as created_at,
               'Varlık Tanıma Sonucu' as description, nr.file_path
        FROM ner_results nr
        WHERE nr.user_id = ? 
        ORDER BY nr.upload_date DESC
    ''', (session['user_id'],))
    ner_result_works = cursor.fetchall()

    # Combine and sort ONLY OUTPUT RESULTS by date, then take top 3
    all_works = (summary_result_works + classification_result_works + ocr_result_works + ner_result_works)
    all_works.sort(key=lambda x: x[4], reverse=True)  # Sort by created_at
    recent_works = all_works[:3]  # Take only the 3 most recent
    


    # Get documents (exclude only pdf_summary results)
    cursor.execute('SELECT * FROM documents WHERE user_id = ? AND category != "pdf_summary" ORDER BY created_at DESC', (session['user_id'],))
    documents = cursor.fetchall()
    
    # Get summary PDFs - now get for current user specifically
    cursor.execute('SELECT * FROM pdf_files WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    summary_pdfs = cursor.fetchall()
    
    # Get classification PDFs - now get for current user specifically
    cursor.execute('SELECT * FROM classification_pdfs WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    classification_pdfs = cursor.fetchall()
    
    # Get Obje ve Karakter Tanıma PDFs - now get for current user specifically  
    cursor.execute('SELECT * FROM ocr_pdfs WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    ocr_pdfs = cursor.fetchall()
    
    # Get Obje ve Karakter Tanıma images - now get for current user specifically
    cursor.execute('SELECT * FROM ocr_images WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    ocr_images = cursor.fetchall()
    
    # Get Varlık Tanıma PDFs - now get for current user specifically
    cursor.execute('SELECT * FROM ner_pdfs WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    ner_pdfs = cursor.fetchall()

    # Get text inputs - now get for current user specifically
    cursor.execute('SELECT * FROM summary_texts WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    summary_texts = cursor.fetchall()
    cursor.execute('SELECT * FROM classification_texts WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    classification_texts = cursor.fetchall()
    cursor.execute('SELECT * FROM ner_texts WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))  # Varlık Tanıma metinleri
    ner_texts = cursor.fetchall()
    
    # Get summary results - now get for current user specifically
    cursor.execute('SELECT * FROM summary_results WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    summary_results = cursor.fetchall()
    
    # Get classification results - now get for current user specifically
    cursor.execute('SELECT * FROM classification_results WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    classification_results = cursor.fetchall()
    
    # Build recent FILES across all types by timestamp (simple and reliable)
    def ts_value(row, idx):
        try:
            raw = row[idx]
            if raw is None:
                return ""
            # Return the string directly for sorting - SQLite timestamps sort correctly as strings
            return str(raw)
        except Exception:
            return ""

    all_files = []
    for row in summary_pdfs:
        # upload_date index 5
        all_files.append((ts_value(row, 5), 'summary_pdf', row))
    for row in classification_pdfs:
        all_files.append((ts_value(row, 5), 'classification_pdf', row))
    for row in ocr_pdfs:
        # Obje ve Karakter Tanıma PDF'leri
        all_files.append((ts_value(row, 5), 'ocr_pdf', row))
    for row in ocr_images:
        # Obje ve Karakter Tanıma resimleri
        all_files.append((ts_value(row, 5), 'ocr_image', row))
    for row in ner_pdfs:
        # Varlık Tanıma PDF'leri
        all_files.append((ts_value(row, 5), 'ner_pdf', row))
    for row in summary_texts:
        all_files.append((ts_value(row, 5), 'summary_text', row))
    for row in classification_texts:
        all_files.append((ts_value(row, 5), 'classification_text', row))
    for row in ner_texts:
        # Varlık Tanıma metinleri
        all_files.append((ts_value(row, 5), 'ner_text', row))
    # NOTE: summary_results ve classification_results dahil DEĞİL - bunlar recent_files'da değil recent_works'te gösterilecek

    # Sort by timestamp desc and take top 3
    all_files.sort(key=lambda x: x[0], reverse=True)
    recent_files = [(t, r) for _, t, r in all_files[:3]]

    conn.close()
    
    return render_template('dashboard.html', 
                         documents=documents, 
                         summary_pdfs=summary_pdfs, 
                         classification_pdfs=classification_pdfs,
                         ocr_pdfs=ocr_pdfs,
                         ocr_images=ocr_images,
                         ner_pdfs=ner_pdfs,
                         summary_texts=summary_texts,
                         classification_texts=classification_texts,
                         ner_texts=ner_texts,
                         recent_files=recent_files,
                         recent_works=recent_works)



@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    try:
        user_id = session['user_id']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Get user info for logging
        cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        username = user[0] if user else 'Unknown'
        
        print(f"=== ACCOUNT DELETION REQUEST ===")
        print(f"User ID: {user_id}")
        print(f"Username: {username}")
        
        # Delete all user files from uploads directory
        upload_dirs = [
            'uploads/summary_pdfs',
            'uploads/classification_pdfs', 
            'uploads/ocr_pdfs',  # Obje ve Karakter Tanıma PDF'leri
            'uploads/ocr_images',  # Obje ve Karakter Tanıma resimleri
            'uploads/ner_pdfs',  # Varlık Tanıma PDF'leri
            'uploads/summary_texts',
            'uploads/classification_texts',
            'uploads/ner_texts',  # Varlık Tanıma metinleri
            'uploads/summary_results',
            'uploads/classification_results',
            'uploads/ocr_results',  # Obje ve Karakter Tanıma sonuçları
            'uploads/ner_results'  # Varlık Tanıma sonuçları
        ]
        
        for upload_dir in upload_dirs:
            if os.path.exists(upload_dir):
                for filename in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, filename)
                    try:
                        # Check if file belongs to this user by checking database
                        if upload_dir == 'uploads/summary_pdfs':
                            cursor.execute('SELECT id FROM pdf_files WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/classification_pdfs':
                            cursor.execute('SELECT id FROM classification_pdfs WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/ocr_pdfs':  # Obje ve Karakter Tanıma PDF'leri
                            cursor.execute('SELECT id FROM ocr_pdfs WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/ocr_images':  # Obje ve Karakter Tanıma resimleri
                            cursor.execute('SELECT id FROM ocr_images WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/ner_pdfs':  # Varlık Tanıma PDF'leri
                            cursor.execute('SELECT id FROM ner_pdfs WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/summary_texts':
                            cursor.execute('SELECT id FROM summary_texts WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/classification_texts':
                            cursor.execute('SELECT id FROM classification_texts WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/ner_texts':  # Varlık Tanıma metinleri
                            cursor.execute('SELECT id FROM ner_texts WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/summary_results':
                            cursor.execute('SELECT id FROM summary_results WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/classification_results':
                            cursor.execute('SELECT id FROM classification_results WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/ocr_results':
                            cursor.execute('SELECT id FROM ocr_results WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        elif upload_dir == 'uploads/ner_results':
                            cursor.execute('SELECT id FROM ner_results WHERE file_path = ? AND user_id = ?', (filename, user_id))
                        
                        if cursor.fetchone():  # File belongs to this user
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")
        
        # Delete all user data from database
        tables_to_clean = [
            'documents', 'pdf_files', 'classification_pdfs', 'ocr_pdfs', 'ocr_images',
            'ner_pdfs', 'summary_texts', 'classification_texts', 'ner_texts',
            'summary_results', 'classification_results', 'ocr_results', 'ner_results'
        ]
        
        for table in tables_to_clean:
            cursor.execute(f'DELETE FROM {table} WHERE user_id = ?', (user_id,))
            deleted_count = cursor.rowcount
            print(f"Deleted {deleted_count} records from {table}")
        
        # Finally delete the user
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        user_deleted = cursor.rowcount
        
        if user_deleted:
            conn.commit()
            print(f"User {username} (ID: {user_id}) successfully deleted")
            
            # Clear session
            session.clear()
            
            conn.close()
            return jsonify({'success': True, 'message': 'Hesap başarıyla silindi'})
        else:
            conn.rollback()
            conn.close()
            return jsonify({'error': 'Kullanıcı bulunamadı'}), 404
            
    except Exception as e:
        print(f"Error deleting account: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return jsonify({'error': f'Hesap silme hatası: {str(e)}'}), 500

@app.route('/summary')
def summary():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('summary.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Generate unique filename while preserving original name
        unique_id = str(uuid.uuid4())
        name, ext = os.path.splitext(file.filename)
        filename = f"{name}_{unique_id}{ext}"
        file_path = os.path.join('uploads', 'summary_pdfs', filename)
        
        # Save file temporarily (not to database yet)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'file_path': file_path,
            'temp_id': unique_id
        })
    
    return jsonify({'error': 'Geçersiz dosya türü'}), 400

@app.route('/summarize_text', methods=['POST'])
def summarize_text_endpoint():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Metin girilmedi'}), 400
    
    if not model_manager.ensure_summarizer_loaded():
        return jsonify({'error': 'Özetleyici model yüklenemedi'}), 500
    
    result = summarize_text(model_manager.summarizer, text)
    if result['success']:
        # Save input text as .txt file under uploads/summary_texts
        try:
            unique_id = str(uuid.uuid4())
            filename = f"summary_text_{unique_id}.txt"
            file_path = os.path.join('uploads', 'summary_texts', filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            file_size = os.path.getsize(file_path)

            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            # Save summary text file metadata
            current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO summary_texts (user_id, original_filename, file_path, file_size, upload_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (session['user_id'], 'Metin Girişi (Özetleme).txt', filename, file_size, current_time))
            # Save summary result to documents
            cursor.execute('''
                INSERT INTO documents (user_id, title, content, category)
                VALUES (?, ?, ?, ?)
            ''', (session['user_id'], 'Metin Özeti', result['summary'], 'text_summary'))
            conn.commit()
            conn.close()
        except Exception:
            pass
   
    return jsonify(result)


@app.route('/save_summary_result', methods=['POST'])
def save_summary_result():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    print(f"Özet kaydetme isteği alındı - User ID: {session['user_id']}")
    data = request.get_json()
    print(f"Gelen veri: {data}")
    summary = data.get('summary', '')
    original_length = data.get('original_length', '')
    summary_length = data.get('summary_length', '')
    compression_rate = data.get('compression_rate', '')
    
    if not summary.strip():
        print("Hata: Özet boş!")
        return jsonify({'error': 'Özet boş olamaz'}), 400
    
    try:
        # Create detailed summary content
        detailed_content = f"""ÖZET RAPORU
{'='*50}

Orijinal Uzunluk: {original_length}
Özet Uzunluğu: {summary_length}  
Sıkıştırma Oranı: {compression_rate}

ÖZET İÇERİĞİ:
{'-'*20}
{summary}

Oluşturulma Tarihi: {get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save summary result as file
        unique_id = str(uuid.uuid4())
        filename = f"ozet_sonucu_{unique_id}.txt"
        file_path = os.path.join('uploads', 'summary_results', filename)
        
        print(f"Dosya yolu: {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(detailed_content)
        
        print(f"Dosya başarıyla kaydedildi: {file_path}")
        
        file_size = os.path.getsize(file_path)
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Find the most recent input file to get its name
        cursor.execute('''
            SELECT original_filename, upload_date FROM summary_texts WHERE user_id = ? 
            UNION ALL
            SELECT original_filename, upload_date FROM pdf_files WHERE user_id = ?
            ORDER BY upload_date DESC LIMIT 1
        ''', (session['user_id'], session['user_id']))
        last_input = cursor.fetchone()
        
        # Create output filename based on input
        if last_input and last_input[0] != 'Metin Girişi (Özetleme).txt':
            # For PDF files, use the PDF name
            input_name = last_input[0]
            if input_name.endswith('.pdf'):
                base_name = input_name[:-4]  # Remove .pdf extension
                output_filename = f'Özet Sonucu - {base_name} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
            else:
                output_filename = f'Özet Sonucu - {input_name} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        else:
            # For text input, use the generic name
            output_filename = f'Özet Sonucu - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        
        # Save to database
        current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Veritabanına kaydediliyor - Dosya adı: {output_filename}")
        cursor.execute('''
            INSERT INTO summary_results (user_id, original_filename, file_path, file_size, summary_content, upload_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], output_filename, filename, file_size, summary, current_time))
        conn.commit()
        conn.close()
        
        print(f"Veritabanına başarıyla kaydedildi!")
        return jsonify({'success': True, 'message': 'Özet başarıyla kaydedildi'})
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return jsonify({'error': f'Kaydetme hatası: {str(e)}'}), 500


@app.route('/save_classification_result', methods=['POST'])
def save_classification_result():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    print(f"=== CLASSIFICATION SAVE DEBUG ===")
    print(f"User ID: {session['user_id']}")
    data = request.get_json()
    print(f"Raw request data: {data}")
    
    main_category = data.get('main_category', '')
    confidence_score = data.get('confidence_score', '')
    categories_html = data.get('categories_html', '')
    
    print(f"Parsed main_category: '{main_category}'")
    print(f"Parsed confidence_score: '{confidence_score}'")
    print(f"Categories HTML length: {len(categories_html)}")
    
    if not main_category.strip():
        print(f"ERROR: Empty main_category - '{main_category}'")
        return jsonify({'error': 'Sınıflandırma sonucu boş olamaz'}), 400
    
    try:
        # Create detailed classification content
        detailed_content = f"""SINIFLANDIRMA RAPORU
{'='*50}

Ana Kategori: {main_category}
Güven Skoru: {confidence_score}

DETAYLI KATEGORILER:
{'-'*20}
{categories_html}

Oluşturulma Tarihi: {get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save classification result as file
        unique_id = str(uuid.uuid4())
        filename = f"siniflandirma_sonucu_{unique_id}.txt"
        file_path = os.path.join('uploads', 'classification_results', filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(detailed_content)
        
        file_size = os.path.getsize(file_path)
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Find the most recent input file to get its name
        cursor.execute('''
            SELECT original_filename, upload_date FROM classification_texts WHERE user_id = ? 
            UNION ALL
            SELECT original_filename, upload_date FROM classification_pdfs WHERE user_id = ?
            ORDER BY upload_date DESC LIMIT 1
        ''', (session['user_id'], session['user_id']))
        last_input = cursor.fetchone()
        
        # Create output filename based on input
        if last_input and last_input[0] != 'Metin Girişi (Sınıflandırma).txt':
            # For PDF files, use the PDF name
            input_name = last_input[0]
            if input_name.endswith('.pdf'):
                base_name = input_name[:-4]  # Remove .pdf extension
                output_filename = f'Sınıflandırma Sonucu - {base_name} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
            else:
                output_filename = f'Sınıflandırma Sonucu - {input_name} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        else:
            # For text input, use the generic name
            output_filename = f'Sınıflandırma Sonucu - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        
        # Save to database
        current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO classification_results (user_id, original_filename, file_path, file_size, main_category, confidence_score, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], output_filename, filename, file_size, main_category, confidence_score, current_time))
        conn.commit()
        conn.close()
        
        print(f"SUCCESS: Classification saved with filename: {output_filename}")
        return jsonify({'success': True, 'message': 'Sınıflandırma sonucu başarıyla kaydedildi'})
        
    except Exception as e:
        return jsonify({'error': f'Kaydetme hatası: {str(e)}'}), 500



@app.route('/classification')
def classification():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('classification.html')


@app.route('/ocr')
def ocr():
    """Obje ve Karakter Tanıma sayfası"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('ocr.html')

@app.route('/ner')
def ner():
    """Varlık Tanıma sayfası"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('ner.html')

@app.route('/previous_works')
def previous_works():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get summary PDFs (summarization works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'summary' as work_type, upload_date as created_at, 
               'Özetleme işlemi' as description, file_path
        FROM pdf_files 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    summary_works = cursor.fetchall()
    
    # Get classification PDFs (classification works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'classification' as work_type, upload_date as created_at,
               'Sınıflandırma işlemi' as description, file_path
        FROM classification_pdfs 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    classification_works = cursor.fetchall()
    
    # Get OCR PDFs (OCR works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ocr_pdf' as work_type, upload_date as created_at,
               'OCR işlemi (PDF)' as description, file_path
        FROM ocr_pdfs 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ocr_pdf_works = cursor.fetchall()
    
    # Get OCR images (OCR works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ocr_image' as work_type, upload_date as created_at,
               'OCR işlemi (Resim)' as description, file_path
        FROM ocr_images 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ocr_image_works = cursor.fetchall()
    
    # Get NER PDFs (NER works)
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ner' as work_type, upload_date as created_at,
               'NER işlemi' as description, file_path
        FROM ner_pdfs 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ner_works = cursor.fetchall()
    
    # Get text input works
    cursor.execute('''
        SELECT id, user_id, original_filename, 'summary_text' as work_type, upload_date as created_at, 
               'Metin Özetleme' as description, file_path
        FROM summary_texts 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    summary_text_works = cursor.fetchall()
    
    cursor.execute('''
        SELECT id, user_id, original_filename, 'classification_text' as work_type, upload_date as created_at,
               'Metin Sınıflandırma' as description, file_path
        FROM classification_texts 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    classification_text_works = cursor.fetchall()
    
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ner_text' as work_type, upload_date as created_at,
               'Metin NER İşlemi' as description, file_path
        FROM ner_texts 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ner_text_works = cursor.fetchall()
    
    # Get summary results
    cursor.execute('''
        SELECT id, user_id, original_filename, 'summary_result' as work_type, upload_date as created_at,
               'Özet Sonucu' as description, file_path
        FROM summary_results 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    summary_result_works = cursor.fetchall()
    
    # Get classification results
    cursor.execute('''
        SELECT id, user_id, original_filename, 'classification_result' as work_type, upload_date as created_at,
               'Sınıflandırma Sonucu' as description, file_path
        FROM classification_results 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    classification_result_works = cursor.fetchall()
    
    # Get OCR results
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ocr_result' as work_type, upload_date as created_at,
               'OCR Sonucu' as description, file_path
        FROM ocr_results 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ocr_result_works = cursor.fetchall()
    
    # Get NER results
    cursor.execute('''
        SELECT id, user_id, original_filename, 'ner_result' as work_type, upload_date as created_at,
               'NER Sonucu' as description, file_path
        FROM ner_results 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (session['user_id'],))
    ner_result_works = cursor.fetchall()
    
    # Combine and sort ONLY OUTPUT RESULTS by date
    all_works = (summary_result_works + classification_result_works + ocr_result_works + ner_result_works)
    all_works.sort(key=lambda x: x[4], reverse=True)  # Sort by created_at
    
    conn.close()
    
    return render_template('previous_works.html', works=all_works)

@app.route('/documents')
def documents():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get documents
    cursor.execute('SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC', (session['user_id'],))
    documents = cursor.fetchall()
    
    # Get summary PDFs
    cursor.execute('SELECT * FROM pdf_files WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    summary_pdfs = cursor.fetchall()
    
    # Get classification PDFs
    cursor.execute('SELECT * FROM classification_pdfs WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    classification_pdfs = cursor.fetchall()
    
    # Get OCR PDFs
    cursor.execute('SELECT * FROM ocr_pdfs WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    ocr_pdfs = cursor.fetchall()
    
    # Get OCR images
    cursor.execute('SELECT * FROM ocr_images WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    ocr_images = cursor.fetchall()
    
    # Get NER PDFs
    cursor.execute('SELECT * FROM ner_pdfs WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    ner_pdfs = cursor.fetchall()

    # Get text inputs
    cursor.execute('SELECT * FROM summary_texts WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    summary_texts = cursor.fetchall()
    cursor.execute('SELECT * FROM classification_texts WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    classification_texts = cursor.fetchall()
    cursor.execute('SELECT * FROM ner_texts WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    ner_texts = cursor.fetchall()
    
    # Get summary results
    cursor.execute('SELECT * FROM summary_results WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    summary_results = cursor.fetchall()
    
    # Get classification results
    cursor.execute('SELECT * FROM classification_results WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    classification_results = cursor.fetchall()
    
    conn.close()
    
    return render_template('documents.html', 
                         documents=documents, 
                         summary_pdfs=summary_pdfs, 
                         classification_pdfs=classification_pdfs,
                         ocr_pdfs=ocr_pdfs,
                         ocr_images=ocr_images,
                         ner_pdfs=ner_pdfs,
                         summary_texts=summary_texts,
                         classification_texts=classification_texts,
                         ner_texts=ner_texts)


@app.route('/my_pdfs')
def my_pdfs():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pdf_files WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    pdf_files = cursor.fetchall()
    conn.close()
    
    return render_template('my_pdfs.html', pdf_files=pdf_files)

@app.route('/my_classification_pdfs')
def my_classification_pdfs():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM classification_pdfs WHERE user_id = ? ORDER BY upload_date DESC', (session['user_id'],))
    classification_pdfs = cursor.fetchall()
    conn.close()
    
    return render_template('my_classification_pdfs.html', classification_pdfs=classification_pdfs)

@app.route('/upload_classification_pdf', methods=['POST'])
def upload_classification_pdf():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    if 'pdf' not in request.files:
        return jsonify({'success': False, 'error': 'PDF dosyası seçilmedi'})
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({'success': False, 'error': 'PDF dosyası seçilmedi'})
    
    if pdf_file and pdf_file.filename.lower().endswith('.pdf'):
        # Generate unique filename while preserving original name
        unique_id = str(uuid.uuid4())
        name, ext = os.path.splitext(pdf_file.filename)
        filename = f"{name}_{unique_id}{ext}"
        file_path = os.path.join('uploads', 'classification_pdfs', filename)
        
        # Save file temporarily (not to database yet)
        pdf_file.save(file_path)
        
        return jsonify({
            'success': True,
            'pdf': {
                'filename': pdf_file.filename,
                'file_path': file_path,
                'temp_id': unique_id
            }
        })
    
    return jsonify({'success': False, 'error': 'Geçersiz dosya türü'})

@app.route('/classify_text', methods=['POST'])
def classify_text_endpoint():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'success': False, 'error': 'Metin boş olamaz'})
    if not model_manager.ensure_classifier_loaded():
        return jsonify({'success': False, 'error': 'Sınıflandırıcı model yüklenemedi'})
    
    result = classify_text(model_manager.classifier, text)
    # Save input text to uploads/classification_texts on success
    if result.get('success'):
        try:
            unique_id = str(uuid.uuid4())
            filename = f"classification_text_{unique_id}.txt"
            file_path = os.path.join('uploads', 'classification_texts', filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            file_size = os.path.getsize(file_path)
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO classification_texts (user_id, original_filename, file_path, file_size, upload_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (session['user_id'], 'Metin Girişi (Sınıflandırma).txt', filename, file_size, current_time))
            conn.commit()
            conn.close()
        except Exception:
            pass
    return jsonify(result)



@app.route('/delete_pdf/<int:pdf_id>', methods=['DELETE'])
def delete_pdf(pdf_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get PDF info
    cursor.execute('SELECT file_path FROM pdf_files WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
    pdf = cursor.fetchone()
    
    if pdf:
        file_path = os.path.join('uploads', 'summary_pdfs', pdf[0])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        cursor.execute('DELETE FROM pdf_files WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'PDF bulunamadı'}), 404

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    # Allow access to PDFs, text files, and common image formats
    allowed_extensions = { '.pdf', '.txt', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff' }
    file_ext = os.path.splitext(filename.lower())[1]
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Geçersiz dosya türü'}), 400
    
    # Check if file exists in any relevant upload directory
    summary_path = os.path.join('uploads', 'summary_pdfs', filename)
    classification_path = os.path.join('uploads', 'classification_pdfs', filename)
    ocr_pdf_path = os.path.join('uploads', 'ocr_pdfs', filename)
    ocr_image_path = os.path.join('uploads', 'ocr_images', filename)
    ner_pdf_path = os.path.join('uploads', 'ner_pdfs', filename)
    summary_text_path = os.path.join('uploads', 'summary_texts', filename)
    classification_text_path = os.path.join('uploads', 'classification_texts', filename)
    ner_text_path = os.path.join('uploads', 'ner_texts', filename)
    summary_result_path = os.path.join('uploads', 'summary_results', filename)
    classification_result_path = os.path.join('uploads', 'classification_results', filename)
    ocr_result_path = os.path.join('uploads', 'ocr_results', filename)
    ner_result_path = os.path.join('uploads', 'ner_results', filename)
    
    if os.path.exists(summary_path):
        return send_file(summary_path, as_attachment=False)
    elif os.path.exists(classification_path):
        return send_file(classification_path, as_attachment=False)
    elif os.path.exists(ocr_pdf_path):
        return send_file(ocr_pdf_path, as_attachment=False)
    elif os.path.exists(ocr_image_path):
        return send_file(ocr_image_path, as_attachment=False)
    elif os.path.exists(ner_pdf_path):
        return send_file(ner_pdf_path, as_attachment=False)
    elif os.path.exists(summary_text_path):
        return send_file(summary_text_path, as_attachment=False)
    elif os.path.exists(classification_text_path):
        return send_file(classification_text_path, as_attachment=False)
    elif os.path.exists(ner_text_path):
        return send_file(ner_text_path, as_attachment=False)
    elif os.path.exists(summary_result_path):
        return send_file(summary_result_path, as_attachment=False)
    elif os.path.exists(classification_result_path):
        return send_file(classification_result_path, as_attachment=False)
    elif os.path.exists(ocr_result_path):
        return send_file(ocr_result_path, as_attachment=False)
    elif os.path.exists(ner_result_path):
        return send_file(ner_result_path, as_attachment=False)
    else:
        return jsonify({'error': 'Dosya bulunamadı'}), 404

@app.route('/delete_document/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get document info
    cursor.execute('SELECT * FROM documents WHERE id = ? AND user_id = ?', (doc_id, session['user_id']))
    document = cursor.fetchone()
    
    if document:
        # Delete from database
        cursor.execute('DELETE FROM documents WHERE id = ? AND user_id = ?', (doc_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'Belge bulunamadı'}), 404

@app.route('/delete_classification_pdf/<int:pdf_id>', methods=['DELETE'])
def delete_classification_pdf(pdf_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get PDF info
    cursor.execute('SELECT file_path FROM classification_pdfs WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
    pdf = cursor.fetchone()
    
    if pdf:
        file_path = os.path.join('uploads', 'classification_pdfs', pdf[0])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        cursor.execute('DELETE FROM classification_pdfs WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'PDF bulunamadı'}), 404

@app.route('/upload_ocr_pdf', methods=['POST'])
def upload_ocr_pdf():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Generate unique filename while preserving original name
        unique_id = str(uuid.uuid4())
        name, ext = os.path.splitext(file.filename)
        filename = f"{name}_{unique_id}{ext}"
        file_path = os.path.join('uploads', 'ocr_pdfs', filename)
        
        # Save file temporarily (not to database yet)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'file_path': file_path,
            'temp_id': unique_id
        })
    
    return jsonify({'error': 'Geçersiz dosya türü'}), 400

@app.route('/upload_ocr_image', methods=['POST'])
def upload_ocr_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    if 'image_file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['image_file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    # Check if file is an image
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file and file_ext in allowed_extensions:
        # Generate unique filename while preserving original name
        unique_id = str(uuid.uuid4())
        name, ext = os.path.splitext(file.filename)
        filename = f"{name}_{unique_id}{ext}"
        file_path = os.path.join('uploads', 'ocr_images', filename)
        
        # Save file temporarily (not to database yet)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'file_path': file_path,
            'temp_id': unique_id
        })
    
    return jsonify({'error': 'Geçersiz dosya türü'}), 400

@app.route('/ocr_pdf', methods=['POST'])
def ocr_pdf():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    pdf_id = data.get('pdf_id', '')
    
    if not pdf_id:
        return jsonify({'error': 'PDF seçilmedi'}), 400
    
    # Find the PDF file in the uploads directory
    upload_dir = os.path.join('uploads', 'ocr_pdfs')
    pdf_path = None
    original_filename = None
    
    for filename in os.listdir(upload_dir):
        if pdf_id in filename:
            pdf_path = os.path.join(upload_dir, filename)
            # Extract original filename by removing UUID part (last part after last underscore)
            name_parts = filename.rsplit('_', 1)  # Split from the right, only once
            original_filename = name_parts[0] + '.pdf' if len(name_parts) > 1 else filename
            break
    
    if not pdf_path:
        return jsonify({'error': 'PDF bulunamadı'}), 404
    
    # Perform real OCR using our processor
    result = ocr_processor.extract_text_from_pdf(pdf_path)
    
    if result['success']:
        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO ocr_pdfs (user_id, original_filename, file_path, file_size, upload_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (session['user_id'], original_filename, filename, os.path.getsize(pdf_path), current_time))
        conn.commit()
        conn.close()
        
        # Clear memory after successful OCR
        clear_memory_after_ocr()
        
        return jsonify({
            'success': True,
            'extracted_text': result['text'],
            'filename': original_filename,
            'method': result['method'],
            'page_count': result['page_count'],
            'text_length': len(result['text'])
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 500

@app.route('/ocr_image', methods=['POST'])
def ocr_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    image_id = data.get('image_id', '')
    
    if not image_id:
        return jsonify({'error': 'Resim seçilmedi'}), 400
    
    # Find the image file in the uploads directory
    upload_dir = os.path.join('uploads', 'ocr_images')
    image_path = None
    original_filename = None
    
    for filename in os.listdir(upload_dir):
        if image_id in filename:
            image_path = os.path.join(upload_dir, filename)
            # Extract original filename by removing UUID part (last part after last underscore)
            name_parts = filename.rsplit('_', 1)  # Split from the right, only once
            original_filename = name_parts[0] + os.path.splitext(filename)[1] if len(name_parts) > 1 else filename
            break
    
    if not image_path:
        return jsonify({'error': 'Resim bulunamadı'}), 404
    
    # Perform real OCR using our processor
    result = ocr_processor.extract_text_from_image(image_path)
    
    if result['success']:
        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO ocr_images (user_id, original_filename, file_path, file_size, upload_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (session['user_id'], original_filename, filename, os.path.getsize(image_path), current_time))
        conn.commit()
        conn.close()
        
        # Clear memory after successful OCR
        clear_memory_after_ocr()
        
        return jsonify({
            'success': True,
            'extracted_text': result['text'],
            'filename': original_filename,
            'text_length': len(result['text'])
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 500

@app.route('/delete_ocr_pdf/<int:pdf_id>', methods=['DELETE'])
def delete_ocr_pdf(pdf_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get PDF info
    cursor.execute('SELECT file_path FROM ocr_pdfs WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
    pdf = cursor.fetchone()
    
    if pdf:
        file_path = os.path.join('uploads', 'ocr_pdfs', pdf[0])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        cursor.execute('DELETE FROM ocr_pdfs WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'PDF bulunamadı'}), 404

@app.route('/delete_ocr_image/<int:image_id>', methods=['DELETE'])
def delete_ocr_image(image_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get image info
    cursor.execute('SELECT file_path FROM ocr_images WHERE id = ? AND user_id = ?', (image_id, session['user_id']))
    image = cursor.fetchone()
    
    if image:
        file_path = os.path.join('uploads', 'ocr_images', image[0])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        cursor.execute('DELETE FROM ocr_images WHERE id = ? AND user_id = ?', (image_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'Resim bulunamadı'}), 404

@app.route('/save_ocr_result', methods=['POST'])
def save_ocr_result():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    print(f"OCR kaydetme isteği alındı - User ID: {session['user_id']}")
    data = request.get_json()
    print(f"Gelen veri: {data}")
    
    ocr_text = data.get('ocr_text', '')
    source_type = data.get('source_type', 'unknown')  # 'pdf' or 'image'
    source_filename = data.get('source_filename', '')
    
    if not ocr_text.strip():
        print("Hata: OCR metni boş!")
        return jsonify({'error': 'OCR metni boş olamaz'}), 400
    
    try:
        # Create detailed OCR content
        detailed_content = f"""OCR RAPORU
{'='*50}

Kaynak Türü: {source_type.upper()}
Kaynak Dosya: {source_filename}
Çıkarılan Metin Uzunluğu: {len(ocr_text)} karakter

OCR METNİ:
{'-'*20}
{ocr_text}

Oluşturulma Tarihi: {get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save OCR result as file
        unique_id = str(uuid.uuid4())
        filename = f"ocr_sonucu_{unique_id}.txt"
        file_path = os.path.join('uploads', 'ocr_results', filename)
        
        print(f"Dosya yolu: {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(detailed_content)
        
        print(f"Dosya başarıyla kaydedildi: {file_path}")
        
        file_size = os.path.getsize(file_path)
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Create output filename based on input
        if source_filename and source_filename != 'Bilinmeyen':
            if source_filename.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                base_name = os.path.splitext(source_filename)[0]
                output_filename = f'OCR Sonucu - {base_name} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
            else:
                output_filename = f'OCR Sonucu - {source_filename} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        else:
            output_filename = f'OCR Sonucu - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        

        
        # Save to database
        current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Veritabanına kaydediliyor - Dosya adı: {output_filename}")
        cursor.execute('''
            INSERT INTO ocr_results (user_id, original_filename, file_path, file_size, ocr_text, source_type, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], output_filename, filename, file_size, ocr_text, source_type, current_time))
        conn.commit()
        conn.close()
        
        print(f"Veritabanına başarıyla kaydedildi!")
        return jsonify({'success': True, 'message': 'OCR sonucu başarıyla kaydedildi'})
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return jsonify({'error': f'Kaydetme hatası: {str(e)}'}), 500

@app.route('/save_ner_result', methods=['POST'])
def save_ner_result():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    print(f"NER kaydetme isteği alındı - User ID: {session['user_id']}")
    data = request.get_json()
    print(f"Gelen veri: {data}")
    
    ner_text = data.get('ner_text', '')
    entities = data.get('entities', [])
    source_type = data.get('source_type', 'unknown')  # 'pdf' or 'text'
    source_filename = data.get('source_filename', '')
    
    if not ner_text.strip():
        print("Hata: NER metni boş!")
        return jsonify({'error': 'NER metni boş olamaz'}), 400
    
    try:
        # Create detailed NER content
        entities_text = ""
        if entities:
            entities_text = "\n".join([f"- {entity['text']} ({entity['type']})" for entity in entities])
        else:
            entities_text = "Hiç varlık bulunamadı."
        
        detailed_content = f"""NER (VARLIK TANIMA) RAPORU
{'='*50}

Kaynak Türü: {source_type.upper()}
Kaynak Dosya: {source_filename}
Analiz Edilen Metin Uzunluğu: {len(ner_text)} karakter
Bulunan Varlık Sayısı: {len(entities) if entities else 0}

BULUNAN VARLIKLAR:
{'-'*20}
{entities_text}

ANALİZ EDİLEN METİN:
{'-'*20}
{ner_text}

Oluşturulma Tarihi: {get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save NER result as file
        unique_id = str(uuid.uuid4())
        filename = f"ner_sonucu_{unique_id}.txt"
        file_path = os.path.join('uploads', 'ner_results', filename)
        
        print(f"Dosya yolu: {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(detailed_content)
        
        print(f"Dosya başarıyla kaydedildi: {file_path}")
        
        file_size = os.path.getsize(file_path)
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Create output filename based on input
        if source_filename and source_filename != 'Bilinmeyen':
            if source_filename.endswith('.pdf'):
                base_name = os.path.splitext(source_filename)[0]
                output_filename = f'NER Sonucu - {base_name} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
            else:
                output_filename = f'NER Sonucu - {source_filename} - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        else:
            output_filename = f'NER Sonucu - {get_turkish_time().strftime("%Y-%m-%d %H:%M")}.txt'
        

        
        # Save to database
        current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Veritabanına kaydediliyor - Dosya adı: {output_filename}")
        import json
        entities_json = json.dumps(entities, ensure_ascii=False) if entities else '[]'
        
        cursor.execute('''
            INSERT INTO ner_results (user_id, original_filename, file_path, file_size, ner_text, entities_json, source_type, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], output_filename, filename, file_size, ner_text, entities_json, source_type, current_time))
        conn.commit()
        conn.close()
        
        print(f"Veritabanına başarıyla kaydedildi!")
        return jsonify({'success': True, 'message': 'NER sonucu başarıyla kaydedildi'})
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return jsonify({'error': f'Kaydetme hatası: {str(e)}'}), 500

@app.route('/upload_ner_pdf', methods=['POST'])
def upload_ner_pdf():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Generate unique filename while preserving original name
        unique_id = str(uuid.uuid4())
        name, ext = os.path.splitext(file.filename)
        filename = f"{name}_{unique_id}{ext}"
        file_path = os.path.join('uploads', 'ner_pdfs', filename)
        
        # Save file temporarily (not to database yet)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'file_path': file_path,
            'temp_id': unique_id
        })
    
    return jsonify({'error': 'Geçersiz dosya türü'}), 400

@app.route('/process_ner_pdf', methods=['POST'])
def process_ner_pdf():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    pdf_id = data.get('pdf_id', '')
    
    if not pdf_id:
        return jsonify({'error': 'PDF seçilmedi'}), 400
    
    # Find the PDF file in the uploads directory
    upload_dir = os.path.join('uploads', 'ner_pdfs')
    pdf_path = None
    original_filename = None
    
    for filename in os.listdir(upload_dir):
        if pdf_id in filename:
            pdf_path = os.path.join(upload_dir, filename)
            # Extract original filename by removing UUID part (last part after last underscore)
            name_parts = filename.rsplit('_', 1)  # Split from the right, only once
            original_filename = name_parts[0] + '.pdf' if len(name_parts) > 1 else filename
            break
    
    if not pdf_path:
        return jsonify({'error': 'PDF bulunamadı'}), 404
    
    # First extract text using OCR
    ocr_result = ocr_processor.extract_text_from_pdf(pdf_path)
    
    if not ocr_result['success']:
        return jsonify({
            'success': False,
            'error': f"Failed to extract text: {ocr_result['error']}"
        }), 500
    
    # Then perform NER analysis
    ner_result = ner_processor.analyze_entities(ocr_result['text'])
    
    if ner_result['success']:
        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO ner_pdfs (user_id, original_filename, file_path, file_size, upload_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (session['user_id'], original_filename, filename, os.path.getsize(pdf_path), current_time))
        conn.commit()
        conn.close()
        
        # Clear memory after OCR and NER processing
        clear_memory_after_ocr()
        
        return jsonify({
            'success': True,
            'entities': ner_result['entities'],
            'entity_count': ner_result['entity_count'],
            'filename': original_filename,
            'text_length': len(ocr_result['text']),
            'extraction_method': ocr_result['method'],
            'text': ocr_result['text']
        })
    else:
        return jsonify({
            'success': False,
            'error': ner_result['error']
        }), 500

@app.route('/delete_ner_pdf/<int:pdf_id>', methods=['DELETE'])
def delete_ner_pdf(pdf_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get PDF info
    cursor.execute('SELECT file_path FROM ner_pdfs WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
    pdf = cursor.fetchone()
    
    if pdf:
        file_path = os.path.join('uploads', 'ner_pdfs', pdf[0])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        cursor.execute('DELETE FROM ner_pdfs WHERE id = ? AND user_id = ?', (pdf_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'PDF bulunamadı'}), 404

@app.route('/process_ner_text', methods=['POST'])
def process_ner_text():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Metin girilmedi'}), 400
    
    # Perform NER analysis directly on text
    ner_result = ner_processor.analyze_entities(text)
    
    if ner_result['success']:
        # Save input text as .txt file under uploads/ner_texts
        try:
            unique_id = str(uuid.uuid4())
            filename = f"ner_text_{unique_id}.txt"
            file_path = os.path.join('uploads', 'ner_texts', filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            file_size = os.path.getsize(file_path)
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO ner_texts (user_id, original_filename, file_path, file_size, upload_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (session['user_id'], 'Metin Girişi (NER).txt', filename, file_size, current_time))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return jsonify({
            'success': True,
            'entities': ner_result['entities'],
            'entity_count': ner_result['entity_count'],
            'text_length': len(text),
            'text': text
        })
    else:
        return jsonify({
            'success': False,
            'error': ner_result['error']
        }), 500

# Enhanced summarization with OCR support
@app.route('/summarize_pdfs', methods=['POST'])
def summarize_pdfs():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    pdf_ids = data.get('pdf_ids', [])
    
    if not pdf_ids:
        return jsonify({'error': 'PDF seçilmedi'}), 400
    
    # Extract text from all PDFs using OCR
    all_text = ""
    pdf_count = 0
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    for temp_id in pdf_ids:
        # Find the PDF file in the uploads directory
        upload_dir = os.path.join('uploads', 'summary_pdfs')
        for filename in os.listdir(upload_dir):
            if temp_id in filename:
                file_path = os.path.join(upload_dir, filename)
                # Extract original filename by removing UUID part (last part after last underscore)
                name_parts = filename.rsplit('_', 1)  # Split from the right, only once
                original_filename = name_parts[0] + '.pdf' if len(name_parts) > 1 else filename
                
                # Extract text using OCR
                ocr_result = ocr_processor.extract_text_from_pdf(file_path)
                
                if ocr_result['success']:
                    all_text += f"\n\n--- {original_filename} ---\n"
                    all_text += ocr_result['text']
                    pdf_count += 1
                    
                    # Save to database
                    current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute('''
                        INSERT INTO pdf_files (user_id, original_filename, file_path, file_size, upload_date)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (session['user_id'], original_filename, filename, os.path.getsize(file_path), current_time))
                
                # Clear memory after each PDF processing
                clear_memory_after_ocr()
                break
    
    conn.commit()
    conn.close()
    
    if not all_text.strip():
        return jsonify({'error': 'PDF\'lerden metin çıkarılamadı'}), 400
    
    # Perform summarization
    if not model_manager.ensure_summarizer_loaded():
        return jsonify({'error': 'Özetleyici model yüklenemedi'}), 500
    
    result = summarize_text(model_manager.summarizer, all_text)
    
    if result['success']:
        # Save summary to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (user_id, title, content, category)
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], f'PDF Özeti ({pdf_count} dosya)', result['summary'], 'pdf_summary'))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'summary': result['summary'],
            'pdf_count': pdf_count,
            'original_length': len(all_text),
            'summary_length': len(result['summary'])
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 500

# Enhanced classification with OCR support
@app.route('/classify_pdfs', methods=['POST'])
def classify_pdfs():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    pdf_ids = data.get('pdf_ids', [])
    
    if not pdf_ids:
        return jsonify({'success': False, 'error': 'PDF seçilmedi'})
    
    # Extract text from all PDFs using OCR
    all_text = ""
    pdf_count = 0
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    for temp_id in pdf_ids:
        # Find the PDF file in the uploads directory
        upload_dir = os.path.join('uploads', 'classification_pdfs')
        for filename in os.listdir(upload_dir):
            if temp_id in filename:
                file_path = os.path.join(upload_dir, filename)
                # Extract original filename by removing UUID part (last part after last underscore)
                name_parts = filename.rsplit('_', 1)  # Split from the right, only once
                original_filename = name_parts[0] + '.pdf' if len(name_parts) > 1 else filename
                
                # Extract text using OCR
                ocr_result = ocr_processor.extract_text_from_pdf(file_path)
                
                if ocr_result['success']:
                    all_text += " " + ocr_result['text']
                    pdf_count += 1
                    
                    # Save to database
                    current_time = get_turkish_time().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute('''
                        INSERT INTO classification_pdfs (user_id, original_filename, file_path, file_size, upload_date)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (session['user_id'], original_filename, filename, os.path.getsize(file_path), current_time))
                
                # Clear memory after each PDF processing
                clear_memory_after_ocr()
                break
    
    conn.commit()
    conn.close()
    
    if not all_text.strip():
        return jsonify({'success': False, 'error': 'PDF\'lerden metin çıkarılamadı'})
    
    # Perform classification
    if not model_manager.ensure_classifier_loaded():
        return jsonify({'success': False, 'error': 'Sınıflandırıcı model yüklenemedi'})
    
    result = classify_text(model_manager.classifier, all_text)
    
    if result['success']:
        return jsonify({
            'success': True,
            'categories': result['categories'],
            'main_category': result['main_category'],
            'confidence': result['confidence'],
            'pdf_count': pdf_count
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        })
def load_model_with_progress(model_name, load_function):
    """Load a single model with progress tracking"""
    print(f"🔄 Starting to load {model_name}...")
    start_time = time.time()
    
    try:
        # Suppress verbose output during model loading
        with SuppressOutput():
            success = load_function()
        
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"✅ {model_name} loaded successfully in {elapsed_time:.2f}s")
            return True, model_name, elapsed_time
        else:
            print(f"❌ Failed to load {model_name}")
            return False, model_name, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Error loading {model_name}: {str(e)}")
        return False, model_name, elapsed_time

def load_models_on_startup():
    """Load all models sequentially before starting the server"""
    
    # Check if models are already loaded (to prevent reloading on Flask restart)
    if (hasattr(model_manager, 'classifier') and model_manager.classifier is not None and
        hasattr(model_manager, 'summarizer') and model_manager.summarizer is not None and
        hasattr(model_manager.ner_processor, 'model_loaded') and model_manager.ner_processor.model_loaded and
        hasattr(model_manager.ocr_processor, 'model_loaded') and model_manager.ocr_processor.model_loaded):
        print("🚀 GovAI models already loaded! Skipping initialization...")
        return True
    
        print("🚀 Initializing GovAI...")
    print("📊 Loading all models sequentially...")
    
    # Define all models to load in order
    models_to_load = [
        ("1. Classifier (XLM-RoBERTa)", model_manager.load_classifier),
        ("2. Summarizer (LLaMA 3.1 8B)", model_manager.load_summarizer),
        ("3. NER Processor (Turkish BERT)", model_manager.load_ner_model),
        ("4. OCR Processor (Qwen2.5-VL)", model_manager.load_ocr_model)
    ]
    
    # Track overall progress
    total_start_time = time.time()
    loaded_models = []
    failed_models = []
    
    # Load models sequentially
    for i, (model_name, load_function) in enumerate(models_to_load, 1):
        print(f"\n{'='*60}")
        print(f"📥 Loading Model {i}/4: {model_name}")
        print(f"{'='*60}")
        
        success, name, elapsed_time = load_model_with_progress(model_name, load_function)
        
        if success:
            loaded_models.append((name, elapsed_time))
        else:
            failed_models.append(name)
        
        # Show progress
        progress = (i / 4) * 100
        print(f"📊 Overall Progress: {progress:.0f}% ({i}/4 models processed)")
    
    # Calculate total time
    total_elapsed_time = time.time() - total_start_time
    
    # Print final summary
    print("\n" + "="*60)
    print(f"📈 FINAL MODEL LOADING SUMMARY")
    print("="*60)
    
    if loaded_models:
        print(f"✅ Successfully loaded models ({len(loaded_models)}/4):")
        for name, elapsed_time in loaded_models:
            print(f"   • {name} ({elapsed_time:.2f}s)")
    
    if failed_models:
        print(f"❌ Failed to load models ({len(failed_models)}/4):")
        for name in failed_models:
            print(f"   • {name}")
    
    print(f"\n⏱️  Total loading time: {total_elapsed_time:.2f}s")
    print("="*60)
    
    # Determine if we should start the server
    all_models_loaded = len(loaded_models) == 4
    
    if all_models_loaded:
        print("🎉 All 4 models loaded successfully! Ready to start server...")
        # Mark models as loaded in the manager
        ModelManager._models_loaded = True
        
        # Clear memory after loading all models
        print("🧹 Clearing memory after model loading...")
        model_manager.cleanup_models()
        
        return True
    else:
        print(f"⚠️  Only {len(loaded_models)}/4 models loaded successfully.")
        print("🤔 Starting server anyway - failed models will be loaded lazily when needed.")
        
        # Clear memory even if some models failed
        print("🧹 Clearing memory after partial model loading...")
        model_manager.cleanup_models()
        
        return True  # Still start the server, failed models will load lazily


@app.route('/delete_work_result/<table_name>/<int:work_id>', methods=['DELETE'])
def delete_work_result(table_name, work_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    if table_name not in ['summary_results', 'classification_results', 'ocr_results', 'ner_results']:
        return jsonify({'error': 'Geçersiz tablo adı'}), 400
    
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        cursor.execute(f'SELECT file_path FROM {table_name} WHERE id = ? AND user_id = ?', (work_id, session['user_id']))
        result = cursor.fetchone()
        
        if not result:
            return jsonify({'error': 'Kayıt bulunamadı'}), 404
        
        file_path = result[0]
        cursor.execute(f'DELETE FROM {table_name} WHERE id = ? AND user_id = ?', (work_id, session['user_id']))
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'Silme işlemi başarısız'}), 404
        
        if table_name == 'summary_results':
            full_file_path = os.path.join('uploads', 'summary_results', file_path)
        elif table_name == 'classification_results':
            full_file_path = os.path.join('uploads', 'classification_results', file_path)
        elif table_name == 'ocr_results':
            full_file_path = os.path.join('uploads', 'ocr_results', file_path)
        elif table_name == 'ner_results':
            full_file_path = os.path.join('uploads', 'ner_results', file_path)
        
        if os.path.exists(full_file_path):
            os.remove(full_file_path)
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Sonuç başarıyla silindi'})
        
    except Exception as e:
        return jsonify({'error': f'Silme hatası: {str(e)}'}), 500


@app.route('/delete_multiple_results', methods=['DELETE'])
def delete_multiple_results():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    work_ids = data.get('work_ids', [])
    
    if not work_ids:
        return jsonify({'error': 'Silinecek öğe bulunamadı'}), 400
    
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        deleted_count = 0
        for work_info in work_ids:
            work_id = work_info['id']
            work_type = work_info['type']
            
            if work_type not in ['summary_result', 'classification_result']:
                continue
                
            table_name = work_type + 's'
            
            cursor.execute(f'SELECT file_path FROM {table_name} WHERE id = ? AND user_id = ?', (work_id, session['user_id']))
            result = cursor.fetchone()
            
            if result:
                file_path = result[0]
                cursor.execute(f'DELETE FROM {table_name} WHERE id = ? AND user_id = ?', (work_id, session['user_id']))
                
                if cursor.rowcount > 0:
                    deleted_count += 1
                    
                    if table_name == 'summary_results':
                        full_file_path = os.path.join('uploads', 'summary_results', file_path)
                    elif table_name == 'classification_results':
                        full_file_path = os.path.join('uploads', 'classification_results', file_path)
                    
                    if os.path.exists(full_file_path):
                        os.remove(full_file_path)
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'{deleted_count} öğe başarıyla silindi'})
        
    except Exception as e:
        return jsonify({'error': f'Silme hatası: {str(e)}'}), 500

if __name__ == '__main__':
    # Load models before starting server
    if load_models_on_startup():
        print("🚀 Starting GovAI server...")
        # Disable reloader to prevent reloading models on file changes
        app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
    else:
        print("Exiting due to initialization failure...")