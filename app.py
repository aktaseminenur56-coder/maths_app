import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import os
from googleapiclient.discovery import build

# ── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Matematik Öğretmen Paneli",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #6c757d; margin: 4px 0 0; }
    .yorum-card {
        border-left: 4px solid #dee2e6;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .stTextArea textarea { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Model & Vektörizer Yükleme ───────────────────────────────────────────────
@st.cache_resource
def modelleri_yukle():
    """Modeli ve vektörizer'ı bir kez yükle, cache'le."""
    model_yolu    = "model/best_model.pkl"
    tfidf_yolu    = "model/tfidf_ngram.pkl"
    duygu_yolu    = "model/duygu_model.pkl"

    modeller = {}

    if os.path.exists(model_yolu):
        with open(model_yolu, "rb") as f:
            modeller["kaygi"] = pickle.load(f)

    if os.path.exists(duygu_yolu):
        with open(duygu_yolu, "rb") as f:
            modeller["duygu"] = pickle.load(f)

    if os.path.exists(tfidf_yolu):
        with open(tfidf_yolu, "rb") as f:
            modeller["tfidf"] = pickle.load(f)

    return modeller


# ── Metin Temizleme ──────────────────────────────────────────────────────────
STOPWORDS = {
    've','veya','ama','ile','bir','bu','şu','o','da','de','mi','mı','mu','mü',
    'için','gibi','kadar','daha','en','çok','var','yok','ben','sen','biz','siz',
    'olan','oldu','olur','değil','ise','ya','ki','ne','hem','bile','sadece',
    'her','hiç','beri','önce','sonra','şimdi','yani','nasıl','neden','hangi',
}

def metin_temizle(metin: str) -> str:
    metin = re.sub(r'http\S+|www\.\S+|@\w+', '', str(metin))
    metin = re.sub(r'[^\w\sşğıöüçŞĞIÖÜÇ.,!?]', ' ', metin, flags=re.UNICODE)
    metin = re.sub(r'(.)\1{2,}', r'\1\1', metin)
    metin = re.sub(r'\s+', ' ', metin).strip()
    return metin

def nlp_temizle(metin: str) -> str:
    metin = metin_temizle(metin).lower()
    metin = re.sub(r'[^\w\sşğıöüç]', ' ', metin, flags=re.UNICODE)
    return ' '.join([k for k in metin.split() if k not in STOPWORDS and len(k) > 2])

def meta_ozellik_cikar(metin: str) -> np.ndarray:
    import unicodedata
    def emoji_say(m):
        return sum(1 for c in str(m) if unicodedata.category(c) in ('So','Sm','Sk') or ord(c) > 127700)
    NEGATIF = ['değil','yok','olmaz','anlamadım','anlamıyorum','zor','kötü','çözemedim','kaygı','korku','stres']

    kelimeler = metin.split()
    return np.array([[
        len(kelimeler),
        len(metin),
        metin.count('?'),
        metin.count('!'),
        emoji_say(metin),
        sum(1 for c in metin if c.isupper()) / max(len(metin), 1),
        int(bool(re.search(r'(.)\1{2,}', metin))),
        int(any(i in metin.lower() for i in NEGATIF)),
        np.mean([len(k) for k in kelimeler]) if kelimeler else 0,
    ]])


# ── YouTube Yardımcı Fonksiyonları ───────────────────────────────────────────
def youtube_video_id_al(url):
    regex = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)(?P<id>[a-zA-Z0-9_-]{11})'
    match = re.match(regex, url)
    if match:
        return match.group('id')
    return None

def youtube_yorumlarini_cek(video_id, max_results=50):
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()
        return [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
    except Exception as e:
        st.error(f"YouTube Canlı Bağlantı Hatası: {e}")
        return []


# ── Tahmin Fonksiyonu ────────────────────────────────────────────────────────
def tahmin_yap(metin: str, modeller: dict) -> dict:
    if not metin.strip() or "tfidf" not in modeller:
        return None

    temiz   = metin_temizle(metin)
    nlp     = nlp_temizle(temiz)
    meta    = meta_ozellik_cikar(temiz)

    tfidf_vec = modeller["tfidf"].transform([nlp])
    kat_bos = csr_matrix(np.zeros((1, 18)))
    X = hstack([tfidf_vec, csr_matrix(meta), kat_bos])

    sonuc = {}

    if "kaygi" in modeller:
        model = modeller["kaygi"]
        try:
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
        except Exception:
            pred  = model.predict(tfidf_vec)[0]
            proba = model.predict_proba(tfidf_vec)[0]

        sonuc["kaygi_var"] = bool(pred)
        sonuc["kaygi_olasilik"] = float(proba[1])

    if "duygu" in modeller:
        try:
            duygu_pred = modeller["duygu"].predict(X)[0]
            duygu_prob = modeller["duygu"].predict_proba(X)
        except Exception:
            duygu_pred = modeller["duygu"].predict(tfidf_vec)[0]
            duygu_prob = modeller["duygu"].predict_proba(tfidf_vec)

        siniflar = modeller["duygu"].classes_
        en_yuksek_idx = duygu_prob[0].argmax()
        sonuc["duygu"] = siniflar[en_yuksek_idx] if hasattr(siniflar[0], '__str__') else str(duygu_pred)
        sonuc["duygu_olasilik"] = float(duygu_prob[0][en_yuksek_idx])
    else:
        if "kaygi_olasilik" in sonuc:
            p = sonuc["kaygi_olasilik"]
            if p > 0.6:
                sonuc["duygu"] = "Olumsuz"
                sonuc["duygu_olasilik"] = p
            elif p < 0.2:
                sonuc["duygu"] = "Olumlu"
                sonuc["duygu_olasilik"] = 1 - p
            else:
                sonuc["duygu"] = "Nötr"
                sonuc["duygu_olasilik"] = 0.7

    soru_var  = '?' in metin
    tesekkur  = any(k in metin.lower() for k in ['teşekkür','sağ ol','eyvallah','helal'])
    elestirir = any(k in metin.lower() for k in ['kötü','berbat','beğenmedim','hayal kırıklığı'])
    if soru_var:
        sonuc["yorum_tipi"] = "Soru / Yardım İsteği"
    elif tesekkur:
        sonuc["yorum_tipi"] = "Övgü / Teşekkür"
    elif elestirir:
        sonuc["yorum_tipi"] = "Eleştiri / Şikayet"
    else:
        sonuc["yorum_tipi"] = "Genel"

    return sonuc


# ── Renk & Emoji Yardımcıları ────────────────────────────────────────────────
DUYGU_RENK = {
    "Olumlu": ("#2ecc71", "😊"),
    "Olumsuz": ("#e74c3c", "😟"),
    "Nötr": ("#95a5a6", "😐"),
    "Karmaşık": ("#f39c12", "🤔"),
}

def kaygi_renk(olasilik):
    if olasilik > 0.6: return "#e74c3c", "🔴", "Yüksek Kaygı"
    if olasilik > 0.3: return "#f39c12", "🟡", "Orta Kaygı"
    return "#2ecc71", "🟢", "Kaygı Yok"


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mathematics.png", width=72)
    st.title("📊 Öğretmen Paneli")
    st.caption("Matematik Yorum Analiz Sistemi")
    st.divider()

    sayfa = st.radio(
        "Sayfa seç",
        ["🔍 Tek Yorum Analizi", "📋 Toplu Analiz", "🎥 Canlı YouTube Analizi", "📈 İstatistikler", "ℹ️ Hakkında"],
        label_visibility="collapsed"
    )
    st.divider()

    modeller = modelleri_yukle()
    if modeller:
        st.success(f"✅ {len(modeller)} model yüklü")
        if "kaygi" in modeller:
            st.caption(f"• Kaygı modeli: LR optimize")
        if "tfidf" in modeller:
            feats = modeller["tfidf"].max_features
            st.caption(f"• TF-IDF: {feats:,} özellik" if feats else f"• TF-IDF: Yüklendi")
    else:
        st.warning("⚠️ Model dosyaları bulunamadı.\n`model/` klasörüne ekle.")
        
    st.divider()
    st.caption("**Model Performansı (Test)**")
    st.caption("ROC-AUC: **0.9895**")
    st.caption("F1(Kaygı): **0.9231**")
    st.caption("Accuracy: **%98.30**")


# ════════════════════════════════════════════════════════════════════
#  SAYFA 1: TEK YORUM ANALİZİ
# ════════════════════════════════════════════════════════════════════
if sayfa == "🔍 Tek Yorum Analizi":
    st.title("🔍 Tek Yorum Analizi")
    st.caption("Bir öğrenci yorumu gir, model duygu ve sınav kaygısını otomatik analiz etsin.")
    st.divider()

    col_input, col_sonuc = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Yorum Gir")
        yorum_metni = st.text_area(
            "Öğrenci yorumu:",
            placeholder="Örnek: Hocam ayt matematik konularını anlatabilir misiniz? Sınav çok yaklaştı, çok endişeleniyorum...",
            height=160,
            label_visibility="collapsed"
        )

        st.caption("**Hızlı test için örnek yorumlar:**")
        ornekler = {
            "😟 Kaygılı": "Hocam ayt sınavına 2 ay kaldı, geometri konularından çok korkuyorum, ne yapmalıyım?",
            "😊 Olumlu": "Hocam çok güzel anlattınız, sayenizde türevi sonunda anladım teşekkürler!",
            "❓ Soru": "Bu sorunun 3. adımını anlayamadım, farklı bir yöntemle çözebilir misiniz?",
            "😐 Nötr": "Video kalitesi iyi, ses net gelmiyor biraz ama genel olarak güzel içerik.",
        }
        cols = st.columns(2)
        for i, (etiket, metin) in enumerate(ornekler.items()):
            if cols[i % 2].button(etiket, use_container_width=True):
                st.session_state["ornek_yorum"] = metin
                st.rerun()

        if "ornek_yorum" in st.session_state:
            yorum_metni = st.session_state["ornek_yorum"]
            del st.session_state["ornek_yorum"]

        analiz_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

    with col_sonuc:
        st.subheader("Analiz Sonuçları")

        if analiz_btn and yorum_metni.strip():
            if not modeller:
                st.error("Model yüklenemedi. `model/` klasörünü kontrol et.")
            else:
                with st.spinner("Analiz ediliyor..."):
                    sonuc = tahmin_yap(yorum_metni, modeller)

                if sonuc:
                    kaygi_rengi, kaygi_emojisi, kaygi_etiketi = kaygi_renk(sonuc.get("kaygi_olasilik", 0))
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {kaygi_rengi}">
                        <p class="metric-value">{kaygi_emojisi} {kaygi_etiketi}</p>
                        <p class="metric-label">Sınav Kaygısı — Güven: %{sonuc.get('kaygi_olasilik',0)*100:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(sonuc.get("kaygi_olasilik", 0), text=f"Kaygı Olasılığı: %{sonuc.get('kaygi_olasilik',0)*100:.1f}")
                    st.divider()

                    c1, c2 = st.columns(2)
                    duygu = sonuc.get("duygu", "Nötr")
                    duygu_rengi, duygu_emojisi = DUYGU_RENK.get(duygu, ("#95a5a6", "😐"))
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid {duygu_rengi}">
                            <p class="metric-value">{duygu_emojisi}</p>
                            <p style="font-weight:600;margin:4px 0">{duygu}</p>
                            <p class="metric-label">Duygu — %{sonuc.get('duygu_olasilik',0)*100:.1f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        tip_emoji = {"Soru / Yardım İsteği":"❓","Övgü / Teşekkür":"🙏","Eleştiri / Şikayet":"⚠️","Genel":"💬"}
                        tip = sonuc.get("yorum_tipi","Genel")
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid #3498db">
                            <p class="metric-value">{tip_emoji.get(tip,'💬')}</p>
                            <p style="font-weight:600;margin:4px 0">{tip}</p>
                            <p class="metric-label">Yorum Tipi</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()
                    if sonuc.get("kaygi_var"):
                        st.info("**📢 Öğretmen Önerisi:** Bu öğrenci sınav kaygısı yaşıyor. Motivasyonu artıracak yanıtlar ve rehberlik faydalı olabilir.")
                    elif duygu == "Olumsuz":
                        st.warning("**📢 Öğretmen Önerisi:** Öğrenci zorlanıyor. Konuyu farklı bir yaklaşımla anlatmayı deneyebilirsiniz.")
                    elif duygu == "Olumlu":
                        st.success("**📢 Öğretmen Önerisi:** Öğrenci memnun! Bu formatı devam ettirebilirsiniz.")
                    else:
                        st.info("**📢 Öğretmen Önerisi:** Genel bir yorum. Özel bir aksiyon gerekmeyebilir.")

        elif analiz_btn:
            st.warning("Lütfen bir yorum girin.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#adb5bd">
                <div style="font-size:3rem">📝</div>
                <p>Sol taraftaki alana bir yorum girin<br>ve <b>Analiz Et</b> butonuna tıklayın.</p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  SAYFA 2: TOPLU ANALİZ
# ════════════════════════════════════════════════════════════════════
elif sayfa == "📋 Toplu Analiz":
    st.title("📋 Toplu Yorum Analizi")
    st.caption("Birden fazla yorumu aynı anda analiz et.")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Yorumları Gir")
        toplu_metin = st.text_area(
            "Her satıra bir yorum girin:",
            height=300,
            placeholder="Hocam çok güzel anlattınız\nSınav çok yaklaştı, geometriden korkuyorum",
            label_visibility="collapsed"
        )

        st.caption("**Veya CSV dosyası yükle:**")
        csv_dosya = st.file_uploader("CSV yükle (yorum sütunu olmalı)", type=["csv"], label_visibility="collapsed")
        toplu_btn = st.button("📊 Toplu Analiz Et", type="primary", use_container_width=True)

    with col2:
        st.subheader("Sonuçlar")

        if toplu_btn:
            yorumlar = []
            if csv_dosya:
                try:
                    df_yuklenen = pd.read_csv(csv_dosya)
                    yorumlar = df_yuklenen['yorum'].dropna().tolist() if 'yorum' in df_yuklenen.columns else df_yuklenen.iloc[:, 0].dropna().tolist()
                    st.success(f"CSV'den {len(yorumlar)} yorum yüklendi.")
                except Exception as e:
                    st.error(f"CSV okunamadı: {e}")
            elif toplu_metin.strip():
                yorumlar = [y.strip() for y in toplu_metin.strip().split('\n') if y.strip()]

            if yorumlar and modeller:
                progress = st.progress(0, text="Analiz ediliyor...")
                sonuclar = []

                for i, yorum in enumerate(yorumlar):
                    s = tahmin_yap(yorum, modeller)
                    if s:
                        sonuclar.append({
                            "Yorum": yorum[:60] + "..." if len(yorum) > 60 else yorum,
                            "Duygu": s.get("duygu", "?"),
                            "Kaygı": "VAR 🔴" if s.get("kaygi_var") else "YOK 🟢",
                            "Kaygı %": f"{s.get('kaygi_olasilik',0)*100:.0f}%",
                            "Tip": s.get("yorum_tipi","?"),
                        })
                    progress.progress((i+1)/len(yorumlar), text=f"{i+1}/{len(yorumlar)} yorum analiz edildi")
                progress.empty()

                if sonuclar:
                    df_sonuc = pd.DataFrame(sonuclar)
                    st.dataframe(df_sonuc, use_container_width=True, height=280)
                    st.divider()
                    
                    toplam = len(df_sonuc)
                    kaygi_sayisi = (df_sonuc["Kaygı"] == "VAR 🔴").sum()
                    olumlu_sayisi = (df_sonuc["Duygu"] == "Olumlu").sum()
                    olumsuz_sayisi = (df_sonuc["Duygu"] == "Olumsuz").sum()

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Toplam", toplam)
                    c2.metric("Kaygılı", kaygi_sayisi, f"%{kaygi_sayisi/toplam*100:.0f}")
                    c3.metric("Olumlu", olumlu_sayisi, f"%{olumlu_sayisi/toplam*100:.0f}")
                    c4.metric("Olumsuz", olumsuz_sayisi, f"%{olumsuz_sayisi/toplam*100:.0f}")

                    csv_indir = df_sonuc.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("⬇️ Sonuçları CSV İndir", data=csv_indir, file_name="yorum_analiz_sonuclari.csv", mime="text/csv", use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  SAYFA 3: CANLI YOUTUBE VİDEO ANALİZİ (YENİ EKLENEN MOTOR 🚀)
# ════════════════════════════════════════════════════════════════════
elif sayfa == "🎥 Canlı YouTube Analizi":
    st.title("🎥 Canlı YouTube Matematik Videosu Analizi")
    st.caption("Bir matematik video linki girin, öğrencilerin anlık kaygı haritasını çıkaralım.")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("YouTube Veri Akışı Yapılandırması")
        video_url = st.text_input("YouTube Video URL'si:", placeholder="https://www.youtube.com/watch?v=...")
        yorum_sayisi = st.slider("Analiz Edilecek Maksimum Yorum Sayısı", min_value=10, max_value=100, value=50, step=10)
        yt_btn = st.button("📊 Canlı Yorumları Çek ve Analiz Et", type="primary", use_container_width=True)

    with col2:
        st.subheader("Canlı Sınıf Analizi")

        if yt_btn:
            if not video_url:
                st.warning("Lütfen geçerli bir YouTube video linki girin.")
            else:
                video_id = youtube_video_id_al(video_url)
                if not video_id:
                    st.error("Hatalı link! Video ID'si tespit edilemedi.")
                else:
                    with st.spinner("YouTube API üzerinden canlı yorumlar çekiliyor..."):
                        canli_yorumlar = youtube_yorumlarini_cek(video_id, max_results=yorum_sayisi)
                    
                    if not canli_yorumlar:
                        st.info("Bu videoya ait canlı yorum bulunamadı veya API bağlantısı kurulamadı.")
                    else:
                        st.success(f"🎉 Canlı Akış Başarılı! {len(canli_yorumlar)} adet öğrenci yorumu işleniyor.")
                        
                        yt_sonuclar = []
                        for y in canli_yorumlar:
                            s = tahmin_yap(y, modeller)
                            if s:
                                yt_sonuclar.append({
                                    "Öğrenci Yorumu": y,
                                    "Kaygı Seviyesi": f"%{s.get('kaygi_olasilik',0)*100:.1f}",
                                    "Durum": "Kaygılı 🚨" if s.get("kaygi_var") else "Sakin 🌱",
                                    "Baskın Duygu": s.get("duygu","Nötr")
                                })
                        
                        df_yt = pd.DataFrame(yt_sonuclar)
                        
                        # Canlı Grafik Alanı
                        kaygili_s = (df_yt["Durum"] == "Kaygılı 🚨").sum()
                        sakin_s = len(df_yt) - kaygili_s
                        
                        fig_yt, ax_yt = plt.subplots(figsize=(4, 3))
                        ax_yt.pie([kaygili_s, sakin_s], labels=['Kaygılı', 'Sakin'], colors=['#e74c3c', '#2ecc71'], autopct='%1.1f%%', startangle=90)
                        ax_yt.axis('equal')
                        st.pyplot(fig_yt, use_container_width=True)
                        
                        st.markdown(f"**🚨 Sınıfta Risk Altındaki Öğrenci Sayısı:** {kaygili_s} / {len(df_yt)}")
                        st.divider()
                        st.markdown("#### 📋 Canlı Yorum Listesi ve Tahminler")
                        st.dataframe(df_yt, use_container_width=True, height=250)


# ════════════════════════════════════════════════════════════════════
#  SAYFA 4: İSTATİSTİKLER
# ════════════════════════════════════════════════════════════════════
elif sayfa == "📈 İstatistikler":
    st.title("📈 Proje İstatistikleri")
    st.divider()

    st.subheader("Model Performansı (Test Seti)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", "%98.30")
    c2.metric("ROC-AUC", "0.9895")
    c3.metric("F1 (Kaygı VAR)", "0.9231")
    c4.metric("False Positive", "0", "Yanlış alarm yok ✅")
    c5.metric("False Negative", "15")

    st.divider()
    st.subheader("Veri Seti Özeti")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        duygu_data = {"Nötr": 5366, "Olumsuz": 287, "Olumlu": 236, "Karmaşık": 2}
        renkler = ["#95a5a6", "#e74c3c", "#2ecc71", "#f39c12"]
        bars = ax.bar(duygu_data.keys(), duygu_data.values(), color=renkler, edgecolor='white')
        for bar, val in zip(bars, duygu_data.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, f'{val:,}', ha='center', fontsize=9)
        ax.set_title('Duygu Dağılımı', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        modeller_adi = ['LR\nbaseline', 'LR\nbalanced', 'Random\nForest', 'XGB\n(BERT)', 'LR\noptimize★']
        roc_skorlar  = [0.9851, 0.9907, 0.9826, 0.9684, 0.9895]
        renkler2 = ['#bdc3c7','#bdc3c7','#bdc3c7','#bdc3c7','#e74c3c']
        b2 = ax2.bar(modeller_adi, roc_skorlar, color=renkler2, edgecolor='white')
        ax2.set_ylim(0.90, 1.01)
        for bar, val in zip(b2, roc_skorlar):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.4f}', ha='center', fontsize=8)
        ax2.set_title('ROC-AUC Karşılaştırması', fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2, use_container_width=True)

    st.divider()
    st.subheader("SHAP — En Önemli Kaygı Kelimeleri")
    shap_data = {"ayt": 0.85, "tyt": 0.78, "lgs": 0.62, "kpss": 0.55, "yks": 0.35, "sınav": 0.30, "neden": 0.22, "kaygı": 0.20}
    fig3, ax3 = plt.subplots(figsize=(9, 3))
    ax3.barh(list(shap_data.keys())[::-1], list(shap_data.values())[::-1], color='#e74c3c', edgecolor='white')
    ax3.set_title('Sınav Kaygısını En Güçlü Tetikleyen Kelimeler', fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    st.pyplot(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  SAYFA 5: HAKKINDA
# ════════════════════════════════════════════════════════════════════
elif sayfa == "ℹ️ Hakkında":
    st.title("ℹ️ Proje Hakkında")
    st.divider()
    st.markdown("""
    ## 🎓 YouTube Matematik Yorumları NLP Projesi
    Bu sistem, online matematik eğitim videolarına gelen öğrenci yorumlarını yapay zeka ile analiz ederek öğretmenlere **gerçek zamanlı geri bildirim** sağlar.
    
    ---
    ### 📊 Proje Özeti
    | Özellik | Detay |
    |---------|-------|
    | Veri seti | 5.891 YouTube matematik yorumu |
    | Hedef | Sınav kaygısı tespiti |
    | Model | Lojistik Regresyon (GridSearch optimize) |
    | Test ROC-AUC | **0.9895** |
    | Test Accuracy | **%98.30** |
    ---
    ### 🔬 Kullanılan Teknolojiler
    - **NLP:** TF-IDF, BERTurk
    - **ML:** Scikit-learn, XGBoost
    - **Açıklanabilirlik:** SHAP
    - **Arayüz:** Streamlit
    """)
