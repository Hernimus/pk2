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

data_nbc = st.session_state.data_normalization.copy()


with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Struktur NBC", "Split Data", "Model & Compute Conditional Probability Tables (CPT)", "Inferensi Probabilistik"],
    )


if selected == "Struktur NBC":
    st.write("Dataset")
    # Step awal ambil data
    st.dataframe(data_nbc.head())

    # Definisikan target
    target_variables = ['GPA_Disc', 'GradeClass']
    
    # Ambil semua fitur selain target
    feature_variables = [col for col in data_nbc.columns if col not in target_variables]
    print(feature_variables)
    
    # Buat Directed Graph
    G_nbc = nx.DiGraph()
    for feature in feature_variables:
        for target in target_variables:
            G_nbc.add_edge(feature, target)
    
    # Visualisasikan
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_nbc)
    nx.draw(G_nbc, pos, with_labels=True, node_size=3000, node_color="lightgreen", font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Struktur Naive Bayes Classifier (Semua Fitur ke Target)", fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

if selected == "Split Data":
    # Pisahkan fitur dan target untuk data latih dan data uji
    X_nbc = data_nbc[feature_variables]
    y_gpa_disc_nbc = data_nbc['GPA_Disc']
    y_grade_class_nbc = data_nbc['GradeClass']
    
    X_train_nbc, X_test_nbc, y_gpa_disc_train_nbc, y_gpa_disc_test_nbc = train_test_split(
        X_nbc, y_gpa_disc_nbc, test_size=0.2, random_state=42, stratify=y_gpa_disc_nbc
    )
    
    _, _, y_grade_class_train_nbc, y_grade_class_test_nbc = train_test_split(
        X_nbc, y_grade_class_nbc, test_size=0.2, random_state=42, stratify=y_grade_class_nbc
    )
    
    # Periksa jumlah kolom pada data latih dan uji
    st.write(f"Dimensi data latih: {X_train_nbc.shape}")
    st.write(f"Dimensi data uji: {X_test_nbc.shape}")
    
    # Pastikan kolom yang digunakan untuk prediksi sama antara latih dan uji
    assert X_train_nbc.shape[1] == X_test_nbc.shape[1], "Jumlah fitur pada data latih dan uji tidak sesuai!"
