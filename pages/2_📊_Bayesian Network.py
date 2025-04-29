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
    page_title="Model Penalaran Probabilistik",
    page_icon="📊",
    layout="wide"
)

data_bn = st.session_state.data_normalization.copy()


with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Struktur Bayesian Network", "Feature dan Split", "Model & Compute Conditional Probability Tables (CPT)", "Fitur Tambahan"],
    )


if selected == "Struktur Bayesian Network":
    st.write("Dataset")
    st.dataframe(data_bn.head())
    
    edges_bn = [
    ('ParentalEducation', 'StudyTimeWeekly_Disc'),
    ('StudyTimeWeekly_Disc', 'GPA_Disc'),
    ('ParentalSupport', 'GPA_Disc'),
    ('Tutoring', 'GPA_Disc'),
    ('Extracurricular', 'GPA_Disc'),
    ('Absences', 'GPA_Disc'),
    ('GPA_Disc', 'GradeClass')
    ]   
    st.session_state.edges_bn = edges_bn

    # Membuat Directed Graph dari edges yang sudah didefinisikan
    model = nx.DiGraph()
    model.add_edges_from(edges_bn)
    

    # 4. Visualisasi struktur jaringan
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(model)  # Posisi node menggunakan layout spring
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, node_color='skyblue',
            font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Bayesian Network Structure for GPA Prediction (with Continuous Variables)", fontsize=14)
    plt.tight_layout() 
    st.pyplot(plt)
    

if selected == "Feature dan Split":
    edges_bn = st.session_state.edges_bn
    # --- 1. Ambil semua node unik dari edges ---
    nodes_in_edges = set()
    for parent, child in edges_bn:
        nodes_in_edges.add(parent)
        nodes_in_edges.add(child)

    # --- 2. Buat list kolom yang mau disimpan ---  
    columns_to_keep = list(nodes_in_edges)
    st.write("Kolom yang dipakai:", columns_to_keep)

    # --- 3. Pisahkan fitur & target dari data_bn ---
    # Target variabel yang akan diprediksi
    target_bn = ["GPA_Disc", "GradeClass"]

    # Fitur berdasarkan edges_bn (drop target supaya tidak bocor)
    features_bn = data_bn[columns_to_keep].drop(columns=target_bn)

    # Target (label)
    target_data = data_bn[target_bn]

    # --- 4. Split data menjadi training dan testing ---
    train_features, test_features, train_target, test_target = train_test_split(
    features_bn,
    target_data,
    test_size=0.2,
    random_state=42,
    stratify=data_bn['GradeClass'] # bisa juga stratify ke GPA_Disc atau GradeClass, pilih sesuai target utama
    )

    # Gabungkan fitur dan target jadi satu dataframe
    train_data_bn = train_features.copy()
    train_data_bn['GPA_Disc'] = train_target['GPA_Disc']
    train_data_bn['GradeClass'] = train_target['GradeClass']

    st.session_state.train_data_bn = train_data_bn

    test_data_bn = test_features.copy()
    test_data_bn['GPA_Disc'] = test_target['GPA_Disc']
    test_data_bn['GradeClass'] = test_target['GradeClass']

    # --- 5. Cek hasil ---
    st.write(f"\nTraining set size: {len(train_features)}")
    st.write(f"Testing set size: {len(test_features)}")
    st.write(f"Fitur digunakan: {train_features.columns.tolist()}")
    st.write(f"Target: {train_target.columns.tolist()}")

if selected == "Model & Compute Conditional Probability Tables (CPT)":
    edges_bn = st.session_state.edges_bn
    train_data_bn = st.session_state.train_data_bn
    # Membuat model jaringan
    model_bn = BayesianNetwork(edges_bn)
    
    # Estimasi parameter (CPT) dengan menggunakan data training
    # Menggunakan estimator BayesianEstimator karena kita bisa mengatur prior
    model_bn.fit(train_data_bn, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)
    
    # --- Menampilkan CPT untuk setiap variabel dalam jaringan ---
    for cpd in model_bn.get_cpds():
        st.write(f"\nCPT untuk {cpd.variable}:")
        st.write(cpd)
