import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="Credit Collectibility Predictor v2", layout="wide")

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_ref():
    # Digunakan untuk menghitung qcut (persentil) secara dinamis
    return pd.read_csv('Data TA (Kredit).csv')

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model('model_xgb_newest_one.json') # Gunakan model terbaru
    return model

# Daftar FCode sesuai LabelEncoder di notebook kamu
fcode_list = ["CA001", "CCB03", "CS0I1", "KJ001", "KJ002", "KJ003", "KJ004", "KJ006", "KJ007", "KK0A5", "KK0B5", "KP001", "KP003", "KP007", "KP07A", "MG001", "MJ008", "RK007"]

# Fungsi Helper untuk Kategorisasi (1-10) berdasarkan data referensi
def get_qcut_label(value, series):
    combined = pd.concat([series, pd.Series([value])], ignore_index=True)
    labels = pd.qcut(combined.rank(method='first'), 10, labels=range(1, 11))
    return int(labels.iloc[-1])

# Fungsi hitung sisa tenor (karena input user adalah tanggal atau bulan)
def calculate_sisa_tenor(target_date_str):
    target = pd.to_datetime('2025-12-31')
    current = pd.to_datetime(target_date_str)
    diff = (current - target).days / 30
    return max(0, diff)

# --- LOAD RESOURCES ---
df_ref = load_ref()
# Hitung sisa tenor referensi untuk qcut sisa tenor
df_ref['Sisa_Tenor_Ref'] = (pd.to_datetime(df_ref['MatDate']) - pd.to_datetime('2025-12-31')).dt.days / 30
df_ref['Sisa_Tenor_Ref'] = df_ref['Sisa_Tenor_Ref'].apply(lambda x: x if x > 0 else 0)

model = load_xgb_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Pilih Menu:", ["🏠 Home", "🔍 Prediksi Tunggal", "📈 Analytics & Insights"])

if menu == "🏠 Home":
    st.title("🏦 Credit Risk Predictor (Updated Version)")
    st.write("Sistem ini telah diperbarui dengan fitur: **effRate, Angsuran, dan Sisa Tenor.**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Fitur Input", "7 Variabel")
    col2.metric("Metode Binning", "Quantile 1-10")
    col3.metric("Algorithm", "XGBoost")

elif menu == "🔍 Prediksi Tunggal":
    st.title("🔍 Prediksi Input Tunggal")
    
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        
        # Kolom Kiri
        f_in = c1.selectbox("Pilih FCode", fcode_list)
        eff_in = c1.number_input("Suku Bunga (effRate %)", min_value=0.0, max_value=100.0, value=11.0)
        os_in = c1.number_input("Nominal Outstanding (OS)", value=50000000.0)
        disb_in = c1.number_input("Nominal Disbursement", value=100000000.0)
        
        # Kolom Kanan
        saldo_in = c2.number_input("Nominal Saldo Rekening", value=5000000.0)
        angsuran_in = c2.number_input("Nominal Angsuran Bulanan", value=2500000.0)
        matdate_in = c2.date_input("Tanggal Jatuh Tempo (Maturity Date)", value=pd.to_datetime('2026-12-31'))
        
        submit = st.form_submit_button("Analisis Risiko Nasabah")
    
    if submit:
        # --- PREPROCESSING ---
        # 1. Encoding FCode
        f_enc = fcode_list.index(f_in) + 1
        
        # 2. Perhitungan Sisa Tenor Mentah
        sisa_tenor_raw = (pd.to_datetime(matdate_in) - pd.to_datetime('2025-12-31')).days / 30
        sisa_tenor_raw = max(0, sisa_tenor_raw)
        
        # 3. Binning 1-10 (Sesuai model training)
        os_cat = get_qcut_label(os_in, df_ref['OS'])
        disb_cat = get_qcut_label(disb_in, df_ref['Disb'])
        saldo_cat = get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
        angsuran_cat = get_qcut_label(angsuran_in, df_ref['Angsuran'])
        tenor_cat = get_qcut_label(sisa_tenor_raw, df_ref['Sisa_Tenor_Ref'])
        
        # --- MATCHING FEATURE ORDER (Sesuai urutan di notebook) ---
        # ['FCode','effRate','OS (Category)','Disb (Category)','Saldo (Category)','Angsuran (Category)','Sisa_Tenor (Category)']
        X_input = pd.DataFrame([[
            f_enc, eff_in, os_cat, disb_cat, saldo_cat, angsuran_cat, tenor_cat
        ]], columns=['FCode', 'effRate', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)', 'Angsuran (Category)', 'Sisa_Tenor (Category)'])
        
        # --- PREDICTION ---
        pred_class = model.predict(X_input)[0] + 1
        
        # Tampilan Hasil
        st.subheader("Hasil Analisis:")
        if pred_class == 1:
            st.success(f"NASABAH LANCAR (Kolektibilitas {pred_class})")
        elif pred_class == 2:
            st.warning(f"DALAM PERHATIAN KHUSUS (Kolektibilitas {pred_class})")
        else:
            st.error(f"NON-PERFORMING LOAN / MACET (Kolektibilitas {pred_class})")
            
        # Tampilkan Nilai Kategori (untuk transparansi data)
        with st.expander("Lihat Detail Kategorisasi (Binning 1-10)"):
            st.write(X_input)

elif menu == "📈 Analytics & Insights":
    st.title("📈 Model Insights")
    
    # Feature Importance
    st.subheader("Variabel Paling Berpengaruh")
    importances = model.feature_importances_
    features = ['FCode', 'effRate', 'OS', 'Disbursement', 'Saldo', 'Angsuran', 'Sisa Tenor']
    
    df_imp = pd.DataFrame({'Fitur': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    fig = px.bar(df_imp, x='Importance', y='Fitur', orientation='h', title="XGBoost Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Catatan Interpretasi:**
    - Nilai Importance yang tinggi menunjukkan fitur tersebut sering digunakan model untuk memisahkan nasabah lancar dan macet.
    - Jika **Saldo** atau **Angsuran** mendominasi, pastikan data input di bagian tersebut akurat.
    """)
