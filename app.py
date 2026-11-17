# VoC Kedai Harapan Djaja — Analisis Sentimen Ulasan (Streamlit Cloud, gratis)
# Versi ringan: TIDAK memuat torch/model besar di server.
# - TF-IDF + LogReg: dimuat lokal dari file (ringan, <1 MB).
# - Zero-shot IndoBERT + LLM Qwen: dipanggil lewat Hugging Face Inference API (serverless, gratis
#   dalam batas pemakaian wajar). Butuh HF_TOKEN yang dimasukkan di Streamlit Secrets — BUKAN di kode.
# - Kalau token belum ada, aplikasi tetap jalan dengan TF-IDF saja.
# Topik (rasa/harga/pelayanan/tempat) BELUM ada — butuh label manual tim (D5) dulu.

import joblib
import streamlit as st

MODEL_ZS = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
MODEL_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
PETA = {"Positive": "Puas", "Neutral": "Netral", "Negative": "Tidak puas"}


@st.cache_resource
def muat_tfidf():
    """Muat model TF-IDF sekali (ringan)."""
    return joblib.load("model_333_tfidf_logreg.pkl")


def dapatkan_client():
    """Client Inference API — None kalau HF_TOKEN belum diisi di Secrets."""
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return None
    from huggingface_hub import InferenceClient
    return InferenceClient(api_key=token)


def analisis_zero_shot(client, teks):
    """Sentimen via Inference API (model jadi, transfer learning)."""
    hasil = client.text_classification(teks, model=MODEL_ZS)
    terbaik = max(hasil, key=lambda h: h["score"])
    return PETA[terbaik["label"]], terbaik["label"], round(float(terbaik["score"]), 3)


def jelaskan_llm(client, ulasan, hasil_zs, hasil_tfidf):
    """Penjelasan + saran tindak lanjut via LLM gratis (Inference API)."""
    pesan = [
        {"role": "system", "content": (
            "Anda asisten Voice of Customer untuk sebuah kedai kopi di Surabaya. "
            "Jawab dalam Bahasa Indonesia, maksimal 3 kalimat, langsung ke inti.")},
        {"role": "user", "content": (
            f"Ulasan pelanggan: \"{ulasan}\"\n"
            f"Hasil analisis sentimen — model IndoBERT: {hasil_zs}; model TF-IDF: {hasil_tfidf}.\n"
            "Tulis: (1) satu kalimat penjelasan mengapa ulasan ini dinilai seperti itu, "
            "(2) satu saran tindak lanjut singkat untuk pihak kedai.")},
    ]
    out = client.chat_completion(model=MODEL_LLM, messages=pesan,
                                 max_tokens=120, temperature=0.7, top_p=0.9)
    return out.choices[0].message.content.strip()


# --- Tampilan ---
st.set_page_config(page_title="VoC Kedai Harapan Djaja — Analisis Sentimen")
st.title("VoC Kedai Harapan Djaja — Analisis Sentimen Ulasan")
st.markdown("""Voice of Customer — Kedai Harapan Djaja Signature, Cabang Gunung Anyar.
Dataset final: 333 ulasan berteks Google Maps (dari 482 ulasan, 16 Nov 2026).

Hasil evaluasi di Colab (label sementara dari bintang):
- Zero-shot IndoBERT: akurasi 0,9592, macro-F1 0,8688, recall Tidak puas 0,7917 (n=294)
- TF-IDF + LogReg: akurasi 0,9509, macro-F1 0,7869, recall Tidak puas 0,4615 (n=326)
- IndoBERT fine-tuning: akurasi 0,9697, macro-F1 0,8669, recall 0,60 — test cuma 66 data (5 keluhan)
""")

teks = st.text_area(
    "Tulis ulasan",
    value="tempat ini bagus tapi agak kurang menarik",
    placeholder="Contoh: makanannya enak tapi pelayanannya lama",
    height=100,
)
pakai_llm = st.checkbox("Tambahkan penjelasan AI (LLM gratis via Inference API)", value=True)

if st.button("Analisis", type="primary"):
    ulasan = (teks or "").strip()
    if not ulasan:
        st.warning("Tulis dulu ulasannya.")
    else:
        # TF-IDF 333 — selalu jalan (lokal, ringan)
        tfidf = muat_tfidf()
        pred = tfidf.predict([ulasan])[0]
        proba = dict(zip(tfidf.classes_, [round(float(x), 3) for x in tfidf.predict_proba([ulasan])[0]]))
        st.subheader("TF-IDF + LogReg (333, baseline)")
        st.write(f"**{pred}** — Puas: {proba.get('Puas', 0)}, Tidak puas: {proba.get('Tidak puas', 0)}")

        # Zero-shot + LLM — butuh HF_TOKEN di Secrets
        client = dapatkan_client()
        if client is None:
            st.info("Model IndoBERT dan LLM belum aktif: tambahkan HF_TOKEN di Streamlit Secrets "
                    "(token gratis dari akun Hugging Face). Hasil TF-IDF di atas tetap bisa dipakai.")
        else:
            try:
                label, label_asli, skor = analisis_zero_shot(client, ulasan)
                st.subheader("Zero-shot IndoBERT (model utama)")
                st.write(f"**{label}** — asli: {label_asli}, skor: {skor}")
                if pakai_llm:
                    with st.spinner("LLM sedang menulis penjelasan..."):
                        penjelasan = jelaskan_llm(
                            client, ulasan,
                            f"{label} (skor {skor})",
                            f"{pred} (Puas {proba.get('Puas', 0)}, Tidak puas {proba.get('Tidak puas', 0)})")
                    st.subheader("Penjelasan & saran tindak lanjut (LLM gratis)")
                    st.write(penjelasan)
            except Exception as e:
                st.error(f"Panggilan Inference API gagal: {e}. "
                         "Hasil TF-IDF di atas tetap valid; coba lagi nanti atau periksa token/limit gratis.")
