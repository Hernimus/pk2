import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
import numpy as np
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from pgmpy.estimators import HillClimbSearch, K2Score, BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Dataset and Variable",
    page_icon="🗃",
    layout="wide"
)

data = pd.read_csv("./data/Student_performance_data_.csv")

with st.sidebar:
    # Cek apakah perlu otomatis memilih menu preprocessing
    if 'auto_select_preprocessing' in st.session_state and st.session_state.auto_select_preprocessing:
        # Set default index ke 1 (PREPROCESSING) agar terpilih otomatis
        selected = option_menu(
            menu_title="DATASET",
            options=["DATASET", "PREPROCESSING"],
            default_index=1  # Index 1 untuk PREPROCESSING
        )
        # Hapus flag setelah digunakan agar tidak mempengaruhi render berikutnya
        del st.session_state['auto_select_preprocessing']
    else:
        # Perilaku normal (default memilih DATASET)
        selected = option_menu(
            menu_title="DATASET",
            options=["DATASET", "PREPROCESSING"],
        )

if selected == "DATASET":
    st.title("DATASET")

    st.header("📋 Preview Data")
    st.dataframe(data.head())
    st.write(data.shape[0], "rows and", data.shape[1], "columns.")

    mem_usage = data.memory_usage(deep=True).sum()
    mem_in_kb = mem_usage / 1024
    mem_in_mb = mem_in_kb / 1024
    st.write(f"Total memory used: {mem_in_kb:.2f} KB ({mem_in_mb:.2f} MB)")

    st.markdown("---")
    st.subheader("Tipe Data")
    st.dataframe(data.dtypes.to_frame(name='Data Type'))

    st.markdown("---")
    st.header("📋 Dataset Describe")
    st.dataframe(data.describe())

    st.markdown("---")
    st.title("👁‍🗨 DETAIL DATASET")

    st.header("🔑 Student ID")
    st.markdown("- **StudentID**: ID unik untuk setiap mahasiswa, mulai dari `1001` hingga `3392`.")

    st.header("👤 Demografi")
    st.markdown("""
    - **Age**: Usia mahasiswa, berkisar antara `15` hingga `18` tahun.  
    - **Gender**: Jenis kelamin:
        - `0`: Laki-laki  
        - `1`: Perempuan  
    - **Ethnicity**: Etnis mahasiswa:
        - `0`: Caucasian  
        - `1`: African American  
        - `2`: Asian  
        - `3`: Other  
    - **ParentalEducation**: Tingkat pendidikan orang tua:
        - `0`: None  
        - `1`: High School  
        - `2`: Some College  
        - `3`: Bachelor's  
        - `4`: Higher  
    """)

    st.header("📖 Kebiasaan Belajar")
    st.markdown("""
    - **StudyTimeWeekly**: Waktu belajar per minggu (jam), mulai dari `0` hingga `20`.  
    - **Absences**: Jumlah ketidakhadiran dalam setahun, dari `0` hingga `30`.  
    - **Tutoring**: Status les tambahan:
        - `0`: Tidak  
        - `1`: Ya  
    """)

    st.header("👨‍👩‍👧‍👦 Keterlibatan Orang Tua")
    st.markdown("""
    - **ParentalSupport**: Tingkat dukungan dari orang tua:
        - `0`: None  
        - `1`: Low  
        - `2`: Moderate  
        - `3`: High  
        - `4`: Very High  
    """)

    st.header("🎯 Kegiatan Ekstrakurikuler")
    st.markdown("""
    - **Extracurricular**: Partisipasi dalam kegiatan ekstrakurikuler:
        - `0`: Tidak  
        - `1`: Ya  
    - **Sports**: Ikut olahraga:  
        - `0`: Tidak  
        - `1`: Ya  
    - **Music**: Ikut kegiatan musik:
        - `0`: Tidak  
        - `1`: Ya  
    - **Volunteering**: Ikut kegiatan sosial/sukarela:
        - `0`: Tidak  
        - `1`: Ya  
    """)

    st.header("📈 Performa Akademik")
    st.markdown("""
    - **GPA**: Rata-rata nilai akhir (Grade Point Average), skala `2.0` hingga `4.0`. Dipengaruhi oleh kebiasaan belajar, dukungan orang tua, dan kegiatan ekstrakurikuler.

    - **GradeClass**: Klasifikasi nilai berdasarkan GPA:
        - `0`: A (GPA ≥ 3.5)  
        - `1`: B (3.0 ≤ GPA < 3.5)  
        - `2`: C (2.5 ≤ GPA < 3.0)  
        - `3`: D (2.0 ≤ GPA < 2.5)  
        - `4`: F (GPA < 2.0)  
    """)

if selected == "PREPROCESSING":
    with st.sidebar:
        sub_selected = option_menu(
            menu_title="",
            options=["Analisis Statistik Deskriptif", "Penanganan Missing Value", "Normalisasi atau diskretisasi variabel", "Analisis Korelasi", "Visualisasi distribusi data untuk memahami pola"],
        )
    if sub_selected == "Analisis Statistik Deskriptif":
        st.header("Analisis Statistik Deskriptif")
        st.header("📋 Dataset Describe")
        st.dataframe(data.describe())
        st.write("### Gender Distribution")
        st.table(data["Gender"].value_counts().to_frame())
        st.write("### Ethnicity Distribution")
        st.write(data["Ethnicity"].value_counts().to_frame())
        st.write("### Parental Education Distribution")
        st.table(data["ParentalEducation"].value_counts().to_frame())
        st.write("### Parental Support Distribution")
        st.table(data["ParentalSupport"].value_counts().to_frame())
        st.write("### Extracurricular Activities Distribution")
        st.table(data["Extracurricular"].value_counts().to_frame())
        st.write("### Sports Participation Distribution")
        st.table(data["Sports"].value_counts().to_frame())
        st.write("### Music Participation Distribution")
        st.table(data["Music"].value_counts().to_frame())
        st.write("### Volunteering Distribution")
        st.table(data["Volunteering"].value_counts().to_frame())
        st.write("### Grade Class Distribution")
        st.table(data["GradeClass"].value_counts().to_frame())

    if sub_selected == "Penanganan Missing Value":     
        st.dataframe(data.isnull().sum().to_frame(name='Missing Values'))
        st.write("Jumlah missing value pada dataset: ", data.isnull().sum().sum())
        st.write("Hasil pengecekan missing value menunjukkan bahwa tidak ada nilai yang hilang di seluruh kolom, sehingga data dalam kondisi lengkap.")
        
    if sub_selected == "Normalisasi atau diskretisasi variabel": 
        # copy data_mapped
        data_normalization = data.copy()
        st.session_state.data_normalization = data_normalization

        # hilangkan whitespace
        data_normalization.columns = data_normalization.columns.str.strip()

        # Tangani GPA dan diskretisasi langsung
        data_normalization['GPA'] = data_normalization['GPA'].apply(lambda x: 1.9 if x < 2.0 else x)

        # Bins dan labels untuk diskretisasi GPA dan StudyTimeWeekly
        gpa_bins = [0, 2.0, 2.5, 3.0, 3.5, float('inf')]  # Bins harus monoton meningkat
        gpa_labels = [4, 3, 2, 1, 0]  # Label untuk setiap bin GPA

        study_bins = [0, 5, 10, 15, float('inf')]  # Bins untuk StudyTimeWeekly
        study_labels = [0, 1, 2, 3]  # Label untuk setiap bin StudyTimeWeekly

        # Terapkan pd.cut untuk diskretisasi
        data_normalization['GPA_Disc'] = pd.cut(data_normalization['GPA'], bins=gpa_bins, labels=gpa_labels, right=False)
        data_normalization['StudyTimeWeekly_Disc'] = pd.cut(data_normalization['StudyTimeWeekly'], bins=study_bins, labels=study_labels, right=False)

        # Drop kolom asli yang sudah didiskritkan
        data_normalization = data_normalization.drop(["GPA", "StudyTimeWeekly"], axis=1)

        # Pisahkan fitur numerik yang perlu didiskritkan
        numerical_features_to_discretize = [col for col in data_normalization.columns if col not in ['GPA_Disc', 'StudyTimeWeekly_Disc']]

        # Diskretisasi sisa fitur yang belum diskrit (gunakan KBinsDiscretizer untuk fitur lainnya)
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        data_normalization[numerical_features_to_discretize] = discretizer.fit_transform(data_normalization[numerical_features_to_discretize])

        # Ubah ke int
        data_normalization[numerical_features_to_discretize] = data_normalization[numerical_features_to_discretize].astype(int)

        # Drop NaN jika ada
        data_normalization = data_normalization.dropna().reset_index(drop=True)

        # Tampilkan hasil diskretisasi
        st.table(data_normalization[['GPA_Disc', 'StudyTimeWeekly_Disc']].head())

        st.table(data_normalization[['Age', 'Absences', 'GPA_Disc', 'StudyTimeWeekly_Disc']].head(5))
    
        st.dataframe(data_normalization[['Age', 'Absences', 'GPA_Disc' , 'StudyTimeWeekly_Disc']].nunique())


    if sub_selected == "Analisis Korelasi": 
        # Salin dataframe
        data_encoded = st.session_state.data_normalization.copy()

        # Ubah kolom kategorikal menjadi numerik
        for col in data_encoded.select_dtypes(include='object').columns:
            data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])

        # Hitung korelasi untuk semua kolom (tanpa StudentID) dengan metode Pearson
        correlation_matrix_all = data_encoded.drop(columns=['StudentID']).corr(method='pearson')

        # Visualisasi korelasi dengan heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Matriks Korelasi Seluruh Variabel (Termasuk Kategorikal)")
        st.pyplot(plt)


    if sub_selected == "Visualisasi distribusi data untuk memahami pola": 
         
        # Scatter plot interaktif untuk dua variabel diskrit
        fig = px.scatter(st.session_state.data_normalization  , x='Age', y='GPA_Disc', color='GradeClass',
                        title="Hubungan antara Age dan GPA Diskrit",
                        labels={'Age': 'Usia', 'GPA_Disc': 'Nilai GPA Diskrit'},
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig)

        
        # Pairplot interaktif antara beberapa variabel diskrit
        fig = px.scatter_matrix(st.session_state.data_normalization  , dimensions=['Age', 'StudyTimeWeekly_Disc', 'Absences', 'GPA_Disc'],
                                color='GradeClass', title="Matrix Hubungan Antar Variabel Diskrit",
                                labels={'Age': 'Usia', 'StudyTimeWeekly_Disc': 'Waktu Belajar Diskrit', 'Absences': 'Absensi', 'GPA_Disc': 'Nilai GPA Diskrit'})
        st.plotly_chart(fig, use_container_width=True)

        # Fitur numerik (kontinu atau telah dinormalisasi)
        numerical_features = ['Age', 'StudyTimeWeekly_Disc', 'Absences', 'GPA_Disc', 'Gender', 'Ethnicity', 'ParentalEducation', 'ParentalSupport',
                                'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GradeClass', 'Tutoring']

        # Visualisasi distribusi untuk fitur numerik (histogram dan boxplot)
        for feature in numerical_features:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.histplot(st.session_state.data_normalization[feature], kde=True)
            plt.title(f'Distribusi {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frekuensi/Kepadatan')

            plt.subplot(1, 2, 2)
            sns.boxplot(y=st.session_state.data_normalization[feature])
            plt.title(f'Box Plot {feature}')
            plt.ylabel(feature)

            plt.tight_layout()
            st.pyplot(plt)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        