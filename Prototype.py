import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import os

# --- CONFIG ---
st.set_page_config(page_title="Credit Collectibility Predictor", layout="wide")

# --- CUSTOM CSS ---
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
    df['MatDate'] = pd.to_datetime(df['MatDate'])
    target_date = pd.to_datetime('2025-12-31')
    df['Sisa_Tenor_Ref'] = (df['MatDate'] - target_date).dt.days / 30
    df['Sisa_Tenor_Ref'] = df['Sisa_Tenor_Ref'].apply(lambda x: x if x > 0 else 0)
    return df

@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    # Build absolute path to the model file
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model_xgb_best.json')
    
    if os.path.exists(model_path):
        model.load_model(model_path)
        return model
    return None

fcode_list = ["CA001", "CCB03", "CS0I1", "KJ001", "KJ002", "KJ003", "KJ004", "KJ006", "KJ007", "KK0A5", "KK0B5", "KP001", "KP003", "KP007", "KP07A", "MG001", "MJ008", "RK007"]

def get_qcut_label(value, series):
    combined = pd.concat([series, pd.Series([value])], ignore_index=True)
    labels = pd.qcut(combined.rank(method='first'), 10, labels=range(1, 11))
    return int(labels.iloc[-1])

# --- SESSION STATE NAVIGASI ---
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"

def set_menu(name):
    st.session_state.menu = name

# --- SIDEBAR ---
with st.sidebar:
    st.title("Sistem Prediksi Kolektibilitas Bank X")
    st.markdown("---")
    if st.button("Home"): set_menu("Home")
    if st.button("Model Training"): set_menu("Model Training")
    if st.button("Prediction & Output"): set_menu("Prediction & Output")
    if st.button("Analytics Dashboard"): set_menu("Analytics Dashboard")
    if st.button("Feature Insights"): set_menu("Feature Insights")
    st.markdown("---")
    st.caption("Dibuat untuk Keperluan Tugas Akhir")

df_ref = load_ref()
model = load_xgb_model()
menu = st.session_state.menu

# ==========================================
# LAMAN 1: HOME 
# ==========================================
if menu == "Home":
    st.title("Sistem Prediksi Kolektibilitas Bank X")
    
    st.markdown("### Langkah Penggunaan Sistem")
    
    col_step1, col_step2 = st.columns(2)
    
    with col_step1:
        st.info("**A. Jika ingin memperbarui Model:**")
        st.markdown("""
        1. Siapkan file dataset nasabah dalam format CSV.
        2. Klik menu **'Model Training'** di sidebar dan unggah file tersebut.
        3. Tunggu sistem melakukan pembersihan data dan penyeimbangan **SMOTETomek**.
        4. Setelah proses selesai, model **XGBoost** baru akan tersimpan secara otomatis.
        """)
        
    with col_step2:
        st.success("**B. Jika ingin melakukan Prediksi:**")
        st.markdown("""
        1. Pilih menu **'Prediction & Output'**.
        2. Gunakan **'Input Tunggal'** untuk pengecekan nasabah secara individu.
        3. Gunakan **'Upload Batch'** untuk mengunggah file data yang ingin diprediksi.
        4. Hasil prediksi tingkat kolektibilitas (1-5) akan langsung ditampilkan.
        """)

    st.divider()
    
    # Ringkasan Status Sistem
    st.subheader("📊 Status Sistem Saat Ini")
    c1, c2 = st.columns(2)
    c1.metric("Data Referensi", f"{len(df_ref):,}")
    c2.metric("Algoritma Utama", "XGBoost")
    
# ==========================================
# LAMAN 2: TRAINING MODEL
# ==========================================
elif menu == "Model Training":
    st.title("Model Training")
    with st.expander("Lihat Format & Kolom CSV yang Dibutuhkan"):
        st.write("Pastikan file CSV Anda memiliki kolom-kolom berikut agar model bisa mengenali fiturnya:")
        
        data_info = {
            "Nama Kolom": ["FCode", "effRate", "OS", "Disb", "Saldo_Rekening", "Angsuran", "MatDate", "Collectibility"],
            "Deskripsi": ["Kode Cabang (Contoh: CA001)", "Bunga Efektif (%)", "Outstanding (Sisa Pinjaman)", "Plafon/Disbursement", "Saldo di Tabungan", "Besar Angsuran bulanan", "Tanggal Jatuh Tempo (YYYY-MM-DD)", "Label Target (Angka 1-5)"],
            "Tipe Data": ["Kategori", "Angka", "Angka", "Angka", "Angka", "Angka", "Tanggal", "Angka (Target)"]
        }
        st.table(pd.DataFrame(data_info))
        
        st.warning("**Penting:** Pastikan tidak ada data kosong (Null/NaN) pada kolom-kolom di atas.")
    st.info("Metode: XGBoost + SMOTETomek (Hybrid Sampling) + 70:30 Split")
    
    up_train = st.file_uploader("Upload Data Baru (CSV)", type="csv")
    
    if up_train:
        df_new = pd.read_csv(up_train)
        required_cols = ["FCode", "effRate", "OS", "Disb", "Saldo_Rekening", "Angsuran", "MatDate", "Collectibility"]
        missing_cols = [c for c in required_cols if c not in df_new.columns]
        
        if missing_cols:
            st.error(f"File ditolak! Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            st.stop()
        
        st.subheader("Pengecekan dan Pembersihan")
        
        null_count = df_new.isnull().sum().sum()
        dup_count = df_new.duplicated().sum()
        
        col_audit1, col_audit2 = st.columns(2)

        is_clean = (null_count == 0 and dup_count == 0)
        
        if null_count > 0:
            col_audit1.warning(f"Ditemukan {null_count} data kosong!")
        else:
            col_audit1.success("Tidak ada data kosong")
            
        if dup_count > 0:
            col_audit2.warning(f"Ditemukan {dup_count} baris duplikat!")
        else:
            col_audit2.success("Tidak ada data duplikat")

        st.divider()

        if is_clean:
            # Jika data sudah bersih, beri tombol warna hijau (primary)
            btn_label = "Mulai Training"
            st.info("Data kamu sudah bersih. Klik tombol di bawah untuk melatih model.")
        else:
            # Jika data kotor, beri peringatan
            btn_label = "Bersihkan Data dan Mulai Training"
            st.warning("Aplikasi akan otomatis menghapus baris kosong atau duplikat sebelum training.")

        if st.button(btn_label, type="primary" if is_clean else "secondary"):
            df_new = df_new.dropna().drop_duplicates()
            
            with st.spinner("Sedang menyeimbangkan dan membersihkan data..."):
                try:
                    # --- 1. DEFINISIKAN X (SOLUSI ERROR 'X IS NOT DEFINED') ---
                    X = pd.DataFrame()
                    
                    # Transformasi FCode ke angka berdasarkan fcode_list
                    X['FCode'] = df_new['FCode'].apply(lambda x: fcode_list.index(x)+1 if x in fcode_list else 1)
                    X['effRate'] = df_new['effRate']
                    
                    # Transformasi variabel numerik ke kategori decile (1-10)
                    X['OS (Category)'] = df_new['OS'].apply(lambda x: get_qcut_label(x, df_ref['OS']))
                    X['Disb (Category)'] = df_new['Disb'].apply(lambda x: get_qcut_label(x, df_ref['Disb']))
                    X['Saldo (Category)'] = df_new['Saldo_Rekening'].apply(lambda x: get_qcut_label(x, df_ref['Saldo_Rekening']))
                    X['Angsuran (Category)'] = df_new['Angsuran'].apply(lambda x: get_qcut_label(x, df_ref['Angsuran']))
                    
                    # Transformasi MatDate ke Sisa Tenor (Category)
                    df_new['MatDate'] = pd.to_datetime(df_new['MatDate'])
                    st_raw = (df_new['MatDate'] - pd.to_datetime('2025-12-31')).dt.days / 30
                    X['Sisa_Tenor (Category)'] = st_raw.apply(lambda x: get_qcut_label(max(0, x), df_ref['Sisa_Tenor_Ref']))
                    
                    # --- 2. DEFINISIKAN TARGET Y ---
                    y = df_new['Collectibility'] - 1 

                    # 3. VALIDASI JUMLAH DATA & SPLIT
                    if len(df_new) < 10:
                        st.error("Data terlalu sedikit untuk melakukan training. Minimal dibutuhkan 10 sampel data.")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                    
                    # 4. PENANGANAN SMOTETOMEK DINAMIS
                    min_samples = y_train.value_counts().min()
                    if min_samples < 2:
                        st.error("Gagal: Ada kelas kolektibilitas yang hanya memiliki 1 data. Butuh minimal 2 data per kelas untuk SMOTE.")
                        st.stop()
                    
                    n_neigh = min(5, min_samples - 1)
                    smote_dist = SMOTE(k_neighbors=n_neigh, random_state=42)
                    smt = SMOTETomek(smote=smote_dist, random_state=42)
                    
                    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

                    dist_sebelum = y_train.value_counts().sort_index()
                    dist_sesudah = pd.Series(y_train_res).value_counts().sort_index()
                    
                    st.session_state.df_summary = pd.DataFrame({
                        "Kolektibilitas": [f"Kolektibilitas {i+1}" for i in range(5)],
                        "Jumlah Sebelum": [dist_sebelum.get(i, 0) for i in range(5)],
                        "Jumlah Sesudah": [dist_sesudah.get(i, 0) for i in range(5)]
                    })
                    
                    # Simpan angka total untuk metrik
                    st.session_state.n_asli = len(X_train)
                    st.session_state.n_hibrida = len(X_train_res)
                    
                    # 5. TRAINING MODEL DENGAN PARAMETER TERBAIK
                    new_model = xgb.XGBClassifier(gamma=0, learning_rate=0.05, max_depth=8, min_child_weight=3, random_state=42)
                    new_model.fit(X_train_res, y_train_res)
                    
                    # --- 6. TAMPILKAN HASIL ---
                    y_pred = new_model.predict(X_test)
                    acc_test = accuracy_score(y_test, y_pred)

                    st.session_state.acc_test = accuracy_score(y_test, y_pred)
                    st.session_state.train_finished = True
                    st.session_state.n_asli = len(X_train)
                    st.session_state.n_hibrida = len(X_train_res)
                    
                    new_model.save_model('model_xgb_best.json')
                    st.cache_resource.clear()

                except Exception as e:
                    st.error(f"Gagal memproses: {e}")

    if 'train_finished' in st.session_state and st.session_state.train_finished:
            st.divider()
            st.success("Pelatihan Model Selesai")
            
            # Menampilkan perbandingan jumlah data sebelum dan sesudah upsampling
            c1, c2, c3 = st.columns(3)
            c1.metric("Akurasi Model", f"{st.session_state.acc_test*100:.2f}%")
            c2.metric("Data Sebelum Upsampling", f"{st.session_state.n_asli} baris")
            c3.metric("Data Sesudah Upsampling", f"{st.session_state.n_hibrida} baris")
            
            st.write("### Detail Perubahan Jumlah Data Per Kelas")
            st.table(st.session_state.df_summary)
            st.info(f"Proses upsampling telah menambah data dari {st.session_state.n_asli} menjadi {st.session_state.n_hibrida} baris untuk menyeimbangkan kelas kolektibilitas.")
                    
# ==========================================
# LAMAN 3: PREDIKSI & OUTPUT
# ==========================================
elif menu == "Prediction & Output":
    if menu == "Prediction & Output":
    if model is None:
        st.warning("⚠️ No trained model found. Please go to **Model Training** first to generate the model.")
    else:
    st.title("Prediksi Kolektibilitas")
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
            
            X = pd.DataFrame([[f_enc, eff_in, os_c, disb_c, saldo_c, angs_c, tenor_c]], 
                             columns=['FCode', 'effRate', 'OS (Category)', 'Disb (Category)', 'Saldo (Category)', 'Angsuran (Category)', 'Sisa_Tenor (Category)'])
            
            pred = model.predict(X)[0] + 1
            
            if pred == 1: bg, txt, status = "#D4EDDA", "#155724", "LANCAR"
            elif pred == 2: bg, txt, status = "#FFF3CD", "#856404", "DALAM PERHATIAN KHUSUS"
            elif pred == 3: bg, txt, status = "#FFE5D0", "#854800", "KURANG LANCAR"
            elif pred == 4: bg, txt, status = "#F8D7DA", "#721C24", "DIRAGUKAN"
            else: bg, txt, status = "#721C24", "#FFFFFF", "MACET / NPL"

            st.markdown(f"""
                <div style="background-color: {bg}; padding: 35px; border-radius: 15px; border: 1px solid {txt}33; text-align: center;">
                    <h1 style="color: {txt}; margin: 0;">Collectibility {pred}</h1>
                    <p style="color: {txt}; font-size: 24px;">{status}</p>
                </div>
            """, unsafe_allow_html=True)

    with t2:
        st.subheader("Upload Batch File (CSV)")
        with st.expander("Lihat Contoh Format Data & Download Template"):
            st.write("Pastikan file CSV Anda memiliki urutan kolom dan format seperti di bawah ini:")
            
            df_contoh = pd.DataFrame({
                "FCode": ["CA001", "KJ001", "KP007"],
                "effRate": [11.5, 12.0, 10.5],
                "OS": [15000000, 200000000, 5000000],
                "Disb": [20000000, 250000000, 10000000],
                "Saldo_Rekening": [1500000, 5000000, 200000],
                "Angsuran": [750000, 4500000, 500000],
                "MatDate": ["2026-12-31", "2027-06-15", "2026-01-20"]
            })
            
            st.table(df_contoh)
            
            csv_template = df_contoh.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Template CSV",
                data=csv_template,
                file_name='template_batch_kolektibilitas.csv',
                mime='text/csv',
            )
       
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
# LAMAN 4: ANALYTICS
# ==========================================
elif menu == "Analytics Dashboard":
    st.title("Dashboard Kredit Nasabah")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total OS", f"Rp {df_ref['OS'].sum()/1e9:.1f} M")
    c2.metric("Total Saldo", f"Rp {df_ref['Saldo_Rekening'].sum()/1e9:.1f} M")
    c3.metric("Total Nasabah", f"{len(df_ref):,}")
    
    st.divider()
    st.subheader("Proyeksi Kolektibilitas")
    
    # Mass prediction untuk dashboard
    df_sample = df_ref.copy()
    X_mass = pd.DataFrame()
    X_mass['FCode'] = df_sample['FCode'].apply(lambda x: fcode_list.index(x) + 1 if x in fcode_list else 1)
    X_mass['effRate'] = df_sample['effRate']
    X_mass['OS (Category)'] = df_sample['OS'].apply(lambda x: get_qcut_label(x, df_ref['OS']))
    X_mass['Disb (Category)'] = df_sample['Disb'].apply(lambda x: get_qcut_label(x, df_ref['Disb']))
    X_mass['Saldo (Category)'] = df_sample['Saldo_Rekening'].apply(lambda x: get_qcut_label(x, df_ref['Saldo_Rekening']))
    X_mass['Angsuran (Category)'] = df_sample['Angsuran'].apply(lambda x: get_qcut_label(x, df_ref['Angsuran']))
    X_mass['Sisa_Tenor (Category)'] = df_sample['Sisa_Tenor_Ref'].apply(lambda x: get_qcut_label(x, df_ref['Sisa_Tenor_Ref']))

    mass_preds = model.predict(X_mass) + 1
    counts = pd.Series(mass_preds).value_counts(normalize=True).sort_index() * 100
    
    cols = st.columns(5)
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#d35400", "#e74c3c"]
    for i in range(5):
        val = counts.get(i+1, 0)
        cols[i].markdown(f"<div style='text-align:center; color:{colors[i]}'><b>Coll {i+1}</b><h3>{val:.1f}%</h3></div>", unsafe_allow_html=True)

# ==========================================
# LAMAN 5: FEATURE INSIGHTS
# ==========================================
elif menu == "Feature Insights":
    st.title("Feature Importance")
    importances = model.feature_importances_
    # Nama fitur disesuaikan dengan 7 variabel
    features = ['FCode', 'effRate', 'OS', 'Disbursement', 'Saldo', 'Angsuran', 'Sisa Tenor']
    df_imp = pd.DataFrame({'Fitur': features, 'Weight': importances}).sort_values(by='Weight', ascending=True)
    
    fig_imp = px.bar(df_imp, x='Weight', y='Fitur', orientation='h', color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig_imp, use_container_width=True)
    st.info(f"Variabel **{df_imp.iloc[-1]['Fitur']}** memiliki pengaruh paling dominan terhadap hasil prediksi.")
