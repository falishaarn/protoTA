import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Credit Collectibility Predictor", layout="wide")

# --- CUSTOM CSS (Sesuai keinginan kamu) ---
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    [data-testid="stSidebar"] .stButton button {
        width: 100%; border-radius: 8px; border: none;
        background-color: transparent; text-align: left;
        padding: 12px 20px; font-size: 16px; color: #31333F;
        transition: 0.3s; margin-bottom: 5px;
    }
    [data-testid="stSidebar"] .stButton button:hover { background-color: #e9ecef; }
    .stMetric { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_ref():
    df = pd.read_csv('Data TA (Kredit).csv')
    # Pre-calculate sisa tenor untuk referensi qcut
    df['MatDate'] = pd.to_datetime(df['MatDate'])
    target_date = pd.to_datetime('2025-12-31')
    df['Sisa_Tenor_Ref'] = (df['MatDate'] - target_date).dt.days / 30
    df['Sisa_Tenor_Ref'] = df['Sisa_Tenor_Ref'].apply(lambda x: x if x > 0 else 0)
    return df

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    # Pastikan nama file sesuai dengan yang kamu save di notebook
    model.load_model('model_xgb_newest_one.json') 
    return model

fcode_list = ["CA001", "CCB03", "CS0I1", "KJ001", "KJ002", "KJ003", "KJ004", "KJ006", "KJ007", "KK0A5", "KK0B5", "KP001", "KP003", "KP007", "KP07A", "MG001", "MJ008", "RK007"]

def get_qcut_label(value, series):
    combined = pd.concat([series, pd.Series([value])], ignore_index=True)
    labels = pd.qcut(combined.rank(method='first'), 10, labels=range(1, 11))
    return int(labels.iloc[-1])

# --- SESSION STATE NAVIGASI ---
if 'menu' not in st.session_state:
    st.session_state.menu = "🏠 Home"

def set_menu(name):
    st.session_state.menu = name

# --- SIDEBAR ---
with st.sidebar:
    st.title("Credit Collectibility Predictor")
    st.markdown("---")
    if st.button("🏠 Home"): set_menu("🏠 Home")
    if st.button("🔍 Prediksi & Output"): set_menu("🔍 Prediction & Output")
    if st.button("📈 Analytics Dashboard"): set_menu("📈 Analytics Dashboard")
    if st.button("🧠 Feature Insights"): set_menu("🧠 Feature Insights")
    st.markdown("---")
    st.caption("Dibuat untuk Keperluan Tugas Akhir")

df_ref = load_ref()
model = load_xgb_model()
menu = st.session_state.menu

# ==========================================
# LAMAN 1: HOME
# ==========================================
if menu == "🏠 Home":
    st.title("🏦 Credit Collectibility Predictor")
    st.write("Navigasikan sistem menggunakan tombol di sidebar untuk memulai analisis.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Sampel Data", f"{len(df_ref):,}")
    with col_b:
        st.metric("Model yang Digunakan", "XGBoost")
    st.info("Sistem ini memprediksi status kolektibilitas (1-5) menggunakan XGBoost yang telah di-tuning.")

# ==========================================
# LAMAN 2: PREDIKSI & OUTPUT
# ==========================================
elif menu == "🔍 Prediction & Output":
    st.title("🔍 Collectibility Predictor")
    t1, t2 = st.tabs(["Input Tunggal", "Upload Batch"])
    
    with t1:
        with st.form("form_p"):
            c1, c2 = st.columns(2)
            f_in = c1.selectbox("Pilih FCode", fcode_list)
            eff_in = c1.number_input("effRate (%)", value=11.0)
            os_in = c1.number_input("Nominal OS", value=10000000.0)
            disb_in = c1.number_input("Nominal Disbursement", value=20000000.0)
            
            saldo_in = c2.number_input("Nominal Saldo", value=1000000.0)
            angs_in = c2.number_input("Nominal Angsuran", value=500000.0)
            mat_in = c2.date_input("Maturity Date", value=pd.to_datetime('2026-12-31'))
            
            btn = st.form_submit_button("Cek Collectibility")
            
        if btn:
            # Preprocessing
            f_enc = fcode_list.index(f_in) + 1
            st_raw = max(0, (pd.to_datetime(mat_in) - pd.to_datetime('2025-12-31')).days / 30)
            
            os_c = get_qcut_label(os_in, df_ref['OS'])
            disb_c = get_qcut_label(disb_in, df_ref['Disb'])
            saldo_c = get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
            angs_c = get_qcut_label(angs_in, df_ref['Angsuran'])
            tenor_c = get_qcut_label(st_raw, df_ref['Sisa_Tenor_Ref'])
            
            # URUTAN HARUS SAMA DENGAN TRAINING
            X = pd.DataFrame([[f_enc, eff_in, os_c, disb_c, saldo_c, angs_c, tenor_c]], 
                             columns=['FCode', 'effRate', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)', 'Angsuran (Category)', 'Sisa_Tenor (Category)'])
            
            pred = model.predict(X)[0] + 1
            
            if pred == 1: bg, txt, status = "#D4EDDA", "#155724", "LANCAR"
            elif pred == 2: bg, txt, status = "#FFF3CD", "#856404", "DALAM PERHATIAN KHUSUS"
            else: bg, txt, status = "#F8D7DA", "#721C24", "NON-PERFORMING LOAN (MACET)"

            st.markdown(f"""
                <div style="background-color: {bg}; padding: 35px; border-radius: 15px; border: 1px solid {txt}33; text-align: center;">
                    <h1 style="color: {txt}; margin: 0;">Collectibility {pred}</h1>
                    <p style="color: {txt}; font-size: 24px;">{status}</p>
                </div>
            """, unsafe_allow_html=True)

    with t2:
        st.subheader("Upload Batch File (CSV)")
        st.info("Pastikan CSV memiliki kolom: `FCode`, `effRate`, `OS`, `Disb`, `Saldo_Rekening`, `Angsuran`, `MatDate` (Format: YYYY-MM-DD)")
        
        up_file = st.file_uploader("Pilih file CSV", type="csv")
        
        if up_file is not None:
            df_up = pd.read_csv(up_file)
            st.write("Preview Data:")
            st.dataframe(df_up.head())
            
            if st.button("Proses Batch Prediksi"):
                results = []
                progress_bar = st.progress(0)
                
                try:
                    for i, row in df_up.iterrows():
                        # 1. Preprocessing FCode
                        f_val = fcode_list.index(row['FCode']) + 1 if row['FCode'] in fcode_list else 1
                        
                        m_date = pd.to_datetime(row['MatDate'])
                        st_raw = max(0, (m_date - pd.to_datetime('2025-12-31')).days / 30)
                        
                        os_c = get_qcut_label(row['OS'], df_ref['OS'])
                        disb_c = get_qcut_label(row['Disb'], df_ref['Disb'])
                        saldo_c = get_qcut_label(row['Saldo_Rekening'], df_ref['Saldo_Rekening'])
                        angs_c = get_qcut_label(row['Angsuran'], df_ref['Angsuran'])
                        tenor_c = get_qcut_label(st_raw, df_ref['Sisa_Tenor_Ref'])
                        
                        X_batch = pd.DataFrame([[
                            f_val, row['effRate'], os_c, disb_c, saldo_c, angs_c, tenor_c
                        ]], columns=['FCode', 'effRate', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)', 'Angsuran (Category)', 'Sisa_Tenor (Category)'])
                        
                        p = model.predict(X_batch)[0] + 1
                        results.append(p)
                        progress_bar.progress((i + 1) / len(df_up))
                    
                    df_up['Prediksi_Collectibility'] = results
                    st.success("Selesai!")
                    st.dataframe(df_up)
                    
                    # Tombol Download
                    st.download_button("📥 Download Hasil", df_up.to_csv(index=False), "hasil_batch.csv", "text/csv")
                
                except Exception as e:
                    st.error(f"Error: Pastikan nama kolom di CSV sudah benar. Detail: {e}")

# ==========================================
# LAMAN 3: ANALYTICS
# ==========================================
elif menu == "📈 Analytics Dashboard":
    st.title("📈 Strategic Risk Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total OS", f"Rp {df_ref['OS'].sum()/1e9:.1f} M")
    c2.metric("Total Saldo", f"Rp {df_ref['Saldo_Rekening'].sum()/1e9:.1f} M")
    c3.metric("Total Nasabah", f"{len(df_ref):,}")
    
    st.divider()
    st.subheader("🎯 Proyeksi Kolektibilitas (Data Sampel)")
    
    # Mass prediction untuk dashboard
    df_sample = df_ref.copy()
    X_mass = pd.DataFrame()
    X_mass['FCode'] = df_sample['FCode'].apply(lambda x: fcode_list.index(x) + 1 if x in fcode_list else 1)
    X_mass['effRate'] = df_sample['effRate']
    X_mass['OS (Category)'] = pd.qcut(df_sample['OS'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
    X_mass['Disb (Category)'] = pd.qcut(df_sample['Disb'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
    X_mass['Saldo (Category)'] = pd.qcut(df_sample['Saldo_Rekening'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
    X_mass['Angsuran (Category)'] = pd.qcut(df_sample['Angsuran'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
    X_mass['Sisa_Tenor (Category)'] = pd.qcut(df_sample['Sisa_Tenor_Ref'].rank(method='first'), 10, labels=range(1, 11)).astype(int)

    mass_preds = model.predict(X_mass) + 1
    counts = pd.Series(mass_preds).value_counts(normalize=True).sort_index() * 100
    
    cols = st.columns(5)
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#d35400", "#e74c3c"]
    for i in range(5):
        val = counts.get(i+1, 0)
        cols[i].markdown(f"<div style='text-align:center; color:{colors[i]}'><b>Coll {i+1}</b><h3>{val:.1f}%</h3></div>", unsafe_allow_html=True)

# ==========================================
# LAMAN 4: FEATURE INSIGHTS
# ==========================================
elif menu == "🧠 Feature Insights":
    st.title("🧠 Feature Importance")
    importances = model.feature_importances_
    # Nama fitur disesuaikan dengan 7 variabel
    features = ['FCode', 'effRate', 'OS', 'Disbursement', 'Saldo', 'Angsuran', 'Sisa Tenor']
    df_imp = pd.DataFrame({'Fitur': features, 'Weight': importances}).sort_values(by='Weight', ascending=True)
    
    fig_imp = px.bar(df_imp, x='Weight', y='Fitur', orientation='h', color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig_imp, use_container_width=True)
    st.info(f"Variabel **{df_imp.iloc[-1]['Fitur']}** memiliki pengaruh paling dominan terhadap hasil prediksi.")
