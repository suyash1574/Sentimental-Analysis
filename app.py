from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
from textblob import TextBlob
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from docx import Document
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Define professional color schemes
COLOR_SCHEMES = {
    'primary': {
        'header': '#1B4F72',  # Deep blue
        'subheader': '#2874A6',  # Medium blue
        'text': '#17202A',  # Dark gray
        'highlight': '#3498DB',  # Light blue
        'accent': '#E74C3C'  # Accent red
    },
    'secondary': {
        'header': '#145A32',  # Deep green
        'subheader': '#196F3D',  # Medium green
        'text': '#17202A',  # Dark gray
        'highlight': '#27AE60',  # Light green
        'accent': '#F39C12'  # Orange accent
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return ' '.join([page.extract_text() for page in reader.pages])

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob with detailed metrics"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    # Tokenize and count words
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    total_words = len(words)
    
    # Count sentiment words
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'happy', 'fantastic', 'perfect', 'brilliant'}
    negative_words = {'bad', 'terrible', 'awful', 'hate', 'disappointed', 'poor', 'worst', 'horrible', 'negative', 'sad'}
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    neutral_count = total_words - positive_count - negative_count
    
    # Get emoji based on sentiment
    if polarity > 0:
        emoji = "😊"
        sentiment = "Positive"
    elif polarity < 0:
        emoji = "😡"
        sentiment = "Negative"
    else:
        emoji = "😐"
        sentiment = "Neutral"
    
    return {
        "sentiment": sentiment,
        "score": polarity,
        "details": f"The text expresses {sentiment.lower()} sentiment.",
        "color": "#28a745" if sentiment == "Positive" else "#dc3545" if sentiment == "Negative" else "#6c757d",
        "emoji": emoji,
        "metrics": {
            "total_words": total_words,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "neutral_words": neutral_count
        }
    }

def generate_pdf_report(data, filename="sentiment_analysis_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Custom style for title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor(COLOR_SCHEMES['primary']['header']),
        spaceAfter=30,
        alignment=1  # Center alignment
    )

    # Custom style for headers
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor(COLOR_SCHEMES['primary']['subheader']),
        spaceAfter=12
    )

    # Add title
    elements.append(Paragraph("Sentiment Analysis Report", title_style))
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {timestamp}", 
                            ParagraphStyle('Timestamp',
                                         parent=styles['Normal'],
                                         textColor=colors.HexColor(COLOR_SCHEMES['primary']['text']),
                                         fontSize=10,
                                         alignment=1,
                                         spaceAfter=20)))

    # Summary section
    elements.append(Paragraph("Analysis Summary", header_style))
    
    # Create summary table with improved styling
    summary_data = [
        ['Metric', 'Value'],
        ['Total Reviews', str(len(data))],
        ['Average Sentiment', f"{sum(d['sentiment_score'] for d in data) / len(data):.2f}"],
        ['Positive Reviews', str(sum(1 for d in data if d['sentiment_score'] > 0.5))],
        ['Negative Reviews', str(sum(1 for d in data if d['sentiment_score'] < 0.5))]
    ]
    
    # Style the summary table
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLOR_SCHEMES['primary']['header'])),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor(COLOR_SCHEMES['primary']['text'])),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#EAECEE')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])
    
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(table_style)
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Detailed results section
    elements.append(Paragraph("Detailed Results", header_style))
    
    for review in data:
        # Create styled review box
        sentiment_color = (colors.HexColor(COLOR_SCHEMES['primary']['highlight']) 
                         if review['sentiment_score'] > 0.5 
                         else colors.HexColor(COLOR_SCHEMES['primary']['accent']))
        
        review_text = f"""
        <para fontSize=10>
        <b>Review:</b> {review['text']}<br/>
        <b>Sentiment Score:</b> <font color={sentiment_color.hexval}>{review['sentiment_score']:.2f}</font>
        </para>
        """
        elements.append(Paragraph(review_text, styles['Normal']))
        elements.append(Spacer(1, 10))

    try:
        doc.build(elements)
        return True
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text based on file type
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif filename.endswith('.docx'):
                    text = extract_text_from_docx(file_path)
                elif filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                
                # Clean up uploaded file
                os.remove(file_path)
            else:
                return jsonify({
                    "error": "Invalid file type",
                    "status": "error"
                }), 400
        else:
            text = request.form.get('text', '')
        
        if not text.strip():
            return jsonify({
                "error": "No text provided",
                "status": "error"
            }), 400
            
        analysis = analyze_sentiment(text)
        
        return jsonify({
            "input": text,
            "analysis": analysis,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    session['history'] = []
    return jsonify({"status": "success"})

@app.route('/clear', methods=['POST'])
def clear():
    session.clear()  # Clear session data if needed
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
