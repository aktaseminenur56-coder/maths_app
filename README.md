# 📊 Matematik Öğretmen Paneli — Kurulum Rehberi

## Klasör Yapısı

```
matematik_app/
├── app.py                    ← Ana Streamlit uygulaması
├── requirements.txt          ← Gerekli kütüphaneler
├── kaggle_model_export.ipynb ← Kaggle'da çalıştır, modelleri export et
├── README.md                 ← Bu dosya
└── model/                    ← Kaggle'dan indirdiğin modeller buraya gelecek
    ├── best_model.pkl
    ├── duygu_model.pkl
    └── tfidf_ngram.pkl
```

---

## Adım Adım Kurulum

### 1. Kaggle'da Modelleri Export Et

1. `kaggle_model_export.ipynb` dosyasını Kaggle'a yükle
2. Dataseti ekle (eminenuraktas1/turkish-math-comments-labeled)
3. **Run All** ile çalıştır
4. **Output** sekmesinden `streamlit_modeller.zip` dosyasını indir

### 2. ZIP'i Çıkart

İndirdiğin `streamlit_modeller.zip` dosyasını çıkart.
İçinde `model/` klasörü çıkacak.
Bu `model/` klasörünü `matematik_app/` içine koy.

### 3. Python Ortamı Kur

Bilgisayarında Python yüklü olmalı.
Komut satırı (CMD veya Terminal) aç:

```bash
# matematik_app klasörüne gir
cd matematik_app

# Sanal ortam oluştur (önerilir)
python -m venv venv

# Windows için aktif et
venv\Scripts\activate

# Mac/Linux için aktif et
source venv/bin/activate

# Kütüphaneleri yükle
pip install -r requirements.txt
```

### 4. Uygulamayı Çalıştır

```bash
streamlit run app.py
```

Tarayıcıda otomatik açılır: `http://localhost:8501`

---

## Hugging Face Spaces'e Yükle (İsteğe Bağlı — Ücretsiz Online Yayın)

1. https://huggingface.co adresine git ve hesap aç (ücretsiz)
2. Sağ üstten **New Space** → **Streamlit** seç
3. `app.py`, `requirements.txt` ve `model/` klasörünü yükle
4. Birkaç dakika bekle — uygulaman online olacak!
5. Paylaşabileceğin bir URL alırsın

---

## Sorun Giderme

**"Model bulunamadı" hatası:**
→ `model/` klasörünün `matematik_app/` içinde olduğundan emin ol

**"Module not found" hatası:**
→ `pip install -r requirements.txt` komutunu tekrar çalıştır

**Port meşgul hatası:**
→ `streamlit run app.py --server.port 8502` dene

---

## Uygulama Özellikleri

- **Tek Yorum Analizi:** Bir yorum gir, anında duygu + kaygı tahmini al
- **Toplu Analiz:** Birden fazla yorum veya CSV dosyası analiz et  
- **İstatistikler:** Model performansı ve veri seti grafikleri
- **Hakkında:** Proje detayları

## Model Performansı

| Metrik | Değer |
|--------|-------|
| Test Accuracy | %98.30 |
| ROC-AUC | 0.9895 |
| F1 (Kaygı VAR) | 0.9231 |
| False Positive | 0 |
