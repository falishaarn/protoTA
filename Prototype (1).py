import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Credit Collectibility Predictor", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    
    /* Style Tombol Navigasi Sidebar */
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background-color: transparent;
        text-align: left;
        padding: 12px 20px;
        font-size: 16px;
        color: #31333F;
        transition: 0.3s;
        margin-bottom: 5px;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #e9ecef;
        border: none;
    }
    
    /* Container Metric Card */
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_ref():
    return pd.read_csv('Data TA (Kredit).csv')

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model('model_xgb.json')
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

# --- SIDEBAR (TANPA BULAT-BULAT) ---
with st.sidebar:
    st.title("Credit Collectibility Predictor")
    st.markdown("---")
    if st.button("🏠 Home"): set_menu("🏠 Home")
    if st.button("🔍 Prediksi & Output"): set_menu("🔍 Prediksi & Output")
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
        st.metric("Model yang Digunakan", "XGBoost Classifier")
    
    st.info("Sistem ini memprediksi status kolektibilitas nasabah (1-5) berdasarkan fitur finansial utama.")

# ==========================================
# LAMAN 2: PREDIKSI & OUTPUT
# ==========================================

elif menu == "🔍 Prediksi & Output":
    st.title("🔍 Prediksi Collectibility")
    t1, t2 = st.tabs(["Input Tunggal", "Upload Batch"])
    
    # --- TAB 1: SINGLE INPUT ---
    with t1:
        with st.form("form_p"):
            c1, c2 = st.columns(2)
            f_in = c1.selectbox("Pilih FCode", fcode_list)
            os_in = c1.number_input("Nominal OS", value=140562406.0)
            disb_in = c2.number_input("Nominal Disbursement", value=210000000.0)
            saldo_in = c2.number_input("Nominal Saldo", value=2530133.0)
            btn = st.form_submit_button("Cek Collectibility")
            
        if btn:
            f_enc = fcode_list.index(f_in) + 1
            os_c = get_qcut_label(os_in, df_ref['OS'])
            disb_c = get_qcut_label(disb_in, df_ref['Disb'])
            saldo_c = get_qcut_label(saldo_in, df_ref['Saldo_Rekening'])
            X = pd.DataFrame([[f_enc, os_c, disb_c, saldo_c]], columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
            pred = model.predict(X)[0] + 1
            
            if pred == 1: bg, txt, status = "#D4EDDA", "#155724", "LANCAR"
            elif pred <= 4: bg, txt, status = "#FFF3CD", "#856404", "DALAM PENGAWASAN (DPK)"
            else: bg, txt, status = "#F8D7DA", "#721C24", "NON-PERFORMING LOAN (NPL) ATAU MACET"

            st.markdown(f"""
                <div style="background-color: {bg}; padding: 35px; border-radius: 15px; border: 1px solid {txt}33; text-align: center;">
                    <p style="color: {txt}; font-size: 18px; font-weight: bold; margin: 0;">HASIL PREDIKSI</p>
                    <h1 style="color: {txt}; font-size: 64px; margin: 10px 0;">Collectibility {pred}</h1>
                    <p style="color: {txt}; font-size: 26px; font-weight: 500; margin: 0;">{status}</p>
                </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: BATCH UPLOAD ---
    with t2:
        st.subheader("Upload Batch File (CSV)")
        st.write("Pastikan file memiliki kolom: `FCode`, `OS`, `Disb`, `Saldo_Rekening`")
        
        up_file = st.file_uploader("Pilih file CSV", type="csv")
        
        if up_file is not None:
            df_up = pd.read_csv(up_file)
            st.write("Preview Data yang Di-upload:")
            st.dataframe(df_up.head())
            
            if st.button("Cek Collectibility"):
                results = []
                # Loading bar untuk estetika
                progress_bar = st.progress(0)
                
                for i, row in df_up.iterrows():
                    # 1. Encoding FCode
                    f_val = fcode_list.index(row['FCode']) + 1 if row['FCode'] in fcode_list else 1
                    # 2. Transformasi ke Kategori (Percentile)
                    os_c = get_qcut_label(row['OS'], df_ref['OS'])
                    disb_c = get_qcut_label(row['Disb'], df_ref['Disb'])
                    saldo_c = get_qcut_label(row['Saldo_Rekening'], df_ref['Saldo_Rekening'])
                    
                    # 3. Predict
                    X_batch = pd.DataFrame([[f_val, os_c, disb_c, saldo_c]], 
                                          columns=['FCode', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)'])
                    p = model.predict(X_batch)[0] + 1
                    results.append(p)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(df_up))
                
                # Tambahkan hasil ke dataframe
                df_up['Prediksi_Collectibility'] = results
                
                st.success(f"Berhasil memproses {len(df_up)} data nasabah!")
                st.dataframe(df_up)
                
                # Fitur Download Hasil
                csv_download = df_up.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_download,
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv"
                )
# ==========================================
# LAMAN 3: ANALYTICS
# ==========================================
elif menu == "📈 Analytics Dashboard":
    st.title("📈 Strategic Risk Dashboard")
    
    # --- 1. METRICS ROW (Tetap) ---
    c1, c2, c3 = st.columns(3)
    total_os = df_ref['OS'].sum()
    total_saldo = df_ref['Saldo_Rekening'].sum()
    c1.metric("Total Outstanding", f"Rp {total_os/1e9:.1f} M")
    c2.metric("Total Dana Nasabah", f"Rp {total_saldo/1e9:.1f} M")
    c3.metric("Basis Nasabah", f"{len(df_ref):,} Orang")

    st.divider()

    # --- 2. PREDIKSI PERSENTASE COLL (PERBAIKAN LOGIKA) ---
    st.subheader("🎯 Credit Collectibility Prediction Percentage")
    
    # Menyiapkan data untuk prediksi massal
    df_batch = df_ref.head(1181).copy()
    
    try:
        X_mass = pd.DataFrame()
        # 1. Encode FCode jadi angka urutan
        X_mass['FCode'] = df_batch['FCode'].apply(lambda x: fcode_list.index(x) + 1 if x in fcode_list else 1)
        
        # 2. Ubah OS, Disb, Saldo jadi kategori 1-10 (Pakai fungsi get_qcut_label yang sudah ada)
        X_mass['OS (Category)'] = df_batch['OS'].apply(lambda x: get_qcut_label(x, df_ref['OS']))
        X_mass['Disb (Category)'] = df_batch['Disb'].apply(lambda x: get_qcut_label(x, df_ref['Disb']))
        X_mass['Saldo (Category)'] = df_batch['Saldo_Rekening'].apply(lambda x: get_qcut_label(x, df_ref['Saldo_Rekening']))
        
        # 3. Jalankan Prediksi
        mass_preds = model.predict(X_mass) + 1
        pred_counts = pd.Series(mass_preds).value_counts(normalize=True).sort_index() * 100
        
        # Tampilkan kotak persentase
        cols = st.columns(5)
        coll_names = ["Coll 1", "Coll 2", "Coll 3", "Coll 4", "Coll 5"]
        coll_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#d35400", "#e74c3c"]
        
        for i in range(5):
            val = pred_counts.get(i+1, 0)
            cols[i].markdown(f"""
                <div style="background-color:{coll_colors[i]}22; border-left:5px solid {coll_colors[i]}; padding:10px; border-radius:5px; text-align:center">
                    <p style="margin:0; font-size:12px; font-weight:bold; color:{coll_colors[i]}">{coll_names[i]}</p>
                    <h3 style="margin:0; color:{coll_colors[i]}">{val:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
        npl_rate = pred_counts.get(3,0) + pred_counts.get(4,0) + pred_counts.get(5,0)
        st.write("")
        st.warning(f"⚠️ **Proyeksi NPL (Non-Performing Loan): {npl_rate:.2f}%**")
        
    except Exception as e:
        st.error(f"Gagal memproses prediksi massal: {e}")

    st.divider()
    
# LAMAN 4: FEATURE INSIGHTS
# ==========================================
elif menu == "🧠 Feature Insights":
    st.title("🧠 Feature Importance Insight")
    
    importances = model.feature_importances_
    features = ['FCode', 'OS (Cat)', 'Disb (Cat)', 'Saldo (Cat)']
    df_imp = pd.DataFrame({'Fitur': features, 'Weight': importances}).sort_values(by='Weight', ascending=True)
    
    col_g, col_t = st.columns([2, 1])
    with col_g:
        fig_imp = px.bar(df_imp, x='Weight', y='Fitur', orientation='h', 
                         title="Bobot Kontribusi Fitur terhadap Prediksi",
                         color_discrete_sequence=['#6c757d'])
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_t:
        st.write("### 💡 Insight Strategis")
        top_feature = df_imp.iloc[-1]['Fitur']
        st.info(f"Fitur **{top_feature}** adalah faktor yang paling memengaruhi keputusan model.")
        st.write(f"""
        Dalam model ini, **{top_feature}** memiliki bobot paling tinggi. 
        Ini mengindikasikan bahwa perubahan pada nilai tersebut berkorelasi kuat dengan perubahan tingkat kolektibilitas nasabah.
        """)
