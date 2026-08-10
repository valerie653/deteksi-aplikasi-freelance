import os
import re
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords', quiet=True)
 
app = Flask(__name__)
 
# ============================================================
# Load Model, TF-IDF, Scaler
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
model  = joblib.load(os.path.join(BASE_DIR, 'model_logreg.pkl'))
tfidf  = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
 
THRESHOLD = 0.60   # threshold optimal dari notebook
 
# ============================================================
# Red Flags & Suspicious Links (sama persis dengan notebook)
# ============================================================
RED_FLAGS = [
    'tanpa modal', 'tanpa pengalaman', 'tanpa syarat',
    'gaji besar', 'gaji tinggi', 'penghasilan besar',
    'komisi harian', 'bonus harian', 'langsung cair',
    'tugas mudah', 'kerja mudah', 'pekerjaan mudah',
    'like rating', 'like dan rating', 'tugas like',
    'deposit', 'top up', 'bayar dulu', 'transfer dulu',
    'verifikasi akun', 'biaya pendaftaran',
    'segera daftar', 'daftar sekarang', 'buruan daftar',
    'slot terbatas', 'kuota terbatas',
    'admin online', 'data entry mudah',
    'no experience', 'no skills required',
    'work from home easy', 'easy money',
    'earn from home', 'make money fast',
    'unlimited income', 'passive income',
    'registration fee', 'payment required',
    'wire transfer', 'western union',
    'act now', 'limited slots', 'urgent hiring',
]
 
SUSPICIOUS_LINKS = [
    't.me', 'wa.me', 'bit.ly', 'tinyurl.com',
    'shorturl.at', 'cutt.ly', 'rebrand.ly',
    'linktr.ee', 'taplink.cc',
]
 
# ============================================================
# Stopwords (sama persis dengan notebook)
# ============================================================
stop_words_en = set(stopwords.words('english'))
stop_words_id = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu',
    'dengan', 'untuk', 'pada', 'adalah', 'akan', 'tidak',
    'dalam', 'juga', 'sudah', 'saya', 'kami', 'kita',
    'bisa', 'ada', 'atau', 'jika', 'serta', 'oleh',
    'lebih', 'dapat', 'saat', 'telah', 'agar', 'hal',
    'lain', 'setelah', 'namun', 'bahwa', 'anda', 'kami',
}
all_stopwords = stop_words_en | stop_words_id
 
# ============================================================
# Fungsi (sama persis dengan notebook)
# ============================================================
def count_red_flags(text):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for flag in RED_FLAGS if flag in text_lower)
 
def detect_suspicious_links(text):
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    return [link for link in SUSPICIOUS_LINKS if link in text_lower]
 
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    for link in SUSPICIOUS_LINKS:
        text = text.replace(link, ' ')
    text = re.sub(r'[@#]\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    return ' '.join(tokens)
 
def prediksi_lowongan(teks):
    rf_count  = count_red_flags(teks)
    links     = detect_suspicious_links(teks)
    has_link  = 1 if links else 0
    flags_det = [f for f in RED_FLAGS if f in teks.lower()]
 
    teks_bersih = preprocess_text(teks)
    X_tf  = tfidf.transform([teks_bersih])
    X_ex  = scaler.transform(csr_matrix([[rf_count, has_link]]))
    X_comb = hstack([X_tf, X_ex])
 
    proba_fraud = float(model.predict_proba(X_comb)[0][1])
    label = 1 if proba_fraud >= THRESHOLD else 0
 
    return {
        'label'       : 'PENIPUAN' if label == 1 else 'ASLI',
        'kode_label'  : label,
        'proba_fraud' : round(proba_fraud * 100, 1),
        'proba_asli'  : round((1 - proba_fraud) * 100, 1),
        'threshold'   : THRESHOLD,
        'rf_count'    : rf_count,
        'flags'       : flags_det,
        'links'       : links,
    }
 
# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    teks = data.get('text', '').strip()
    if not teks:
        return jsonify({'error': 'Teks kosong'}), 400
    hasil = prediksi_lowongan(teks)
    return jsonify(hasil)
 
if __name__ == '__main__':
    app.run(debug=True, port=5000)