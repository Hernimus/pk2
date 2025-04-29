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


st.write("Dataset")
# Step awal ambil data
st.dataframe(data_nbc.head())

# Definisikan target
target_variables = ['GPA_Disc', 'GradeClass']

# Ambil semua fitur selain target
feature_variables = [col for col in data_nbc.columns if col not in target_variables]
st.write(feature_variables)
st.session_state.feature_variables = feature_variables

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


# Pisahkan fitur dan target untuk data latih dan data uji
X_nbc = data_nbc[feature_variables]
y_gpa_disc_nbc = data_nbc['GPA_Disc']
y_grade_class_nbc = data_nbc['GradeClass']

# Split data untuk GPA_Disc
X_train_nbc, X_test_nbc, y_gpa_disc_train_nbc, y_gpa_disc_test_nbc = train_test_split(
    X_nbc, y_gpa_disc_nbc, test_size=0.2, random_state=42, stratify=y_gpa_disc_nbc
)

# Split data untuk GradeClass (menggunakan X_nbc yang sama)
_, _, y_grade_class_train_nbc, y_grade_class_test_nbc = train_test_split(
    X_nbc, y_grade_class_nbc, test_size=0.2, random_state=42, stratify=y_grade_class_nbc
)


# Periksa jumlah kolom pada data latih dan uji
st.write(f"Dimensi data latih: {X_train_nbc.shape}")
st.write(f"Dimensi data uji: {X_test_nbc.shape}")

# Pastikan kolom yang digunakan untuk prediksi sama antara latih dan uji
assert X_train_nbc.shape[1] == X_test_nbc.shape[1], "Jumlah fitur pada data latih dan uji tidak sesuai!"


# Model untuk GPA_Disc
model_gpa_disc_nbc = CategoricalNB()
model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)

# Model untuk GradeClass
model_grade_class_nbc = CategoricalNB()
model_grade_class_nbc.fit(X_train_nbc, y_grade_class_train_nbc)

st.write(model_gpa_disc_nbc.fit)


# Model untuk GPA_Disc
model_gpa_disc_nbc = CategoricalNB()
model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)

# Model untuk GradeClass
model_grade_class_nbc = CategoricalNB()
model_grade_class_nbc.fit(X_train_nbc, y_grade_class_train_nbc)

# CPT untuk GPA_Disc
print("Tabel Probabilitas Fitur untuk GPA_Disc:")
for class_idx, class_log_prob in enumerate(model_gpa_disc_nbc.feature_log_prob_):
    print(f"\nKelas {class_idx}:")
    probs = np.exp(class_log_prob)  # balik dari log-prob ke prob
    for idx, prob in enumerate(probs):
        if isinstance(prob, np.ndarray):
            for cat_idx, p in enumerate(prob):
                print(f"  Feature {idx} - Category {cat_idx}: Probabilitas: {p:.4f}")
        else:
            print(f"  Feature {idx}: Probabilitas: {prob:.4f}")

# CPT untuk GradeClass
print("\nTabel Probabilitas Fitur untuk GradeClass:")
for class_idx, class_log_prob in enumerate(model_grade_class_nbc.feature_log_prob_):
    print(f"\nKelas {class_idx}:")
    probs = np.exp(class_log_prob)
    for idx, prob in enumerate(probs):
        if isinstance(prob, np.ndarray):
            for cat_idx, p in enumerate(prob):
                print(f"  Feature {idx} - Category {cat_idx}: Probabilitas: {p:.4f}")
        else:
            print(f"  Feature {idx}: Probabilitas: {prob:.4f}")

st.subheader("Inferensi Probabilitas")
# Hapus StudentID
X_train_nbc = X_train_nbc.drop(columns=['StudentID'])
X_test_nbc = X_test_nbc.drop(columns=['StudentID'])

# Gunakan KBinsDiscretizer untuk mendiskritisasi (opsional sebenarnya, kalau mau)
discretizer_nbc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# Transformasi semua fitur (tanpa StudentID)
X_train_nbc_transformed = discretizer_nbc.fit_transform(X_train_nbc)
X_test_nbc_transformed = discretizer_nbc.transform(X_test_nbc)

# Model untuk GPA_Disc
model_gpa_disc_nbc_inf = CategoricalNB()
model_gpa_disc_nbc_inf.fit(X_train_nbc_transformed, y_gpa_disc_train_nbc)

# Model untuk GradeClass
model_grade_class_nbc_inf = CategoricalNB()
model_grade_class_nbc_inf.fit(X_train_nbc_transformed, y_grade_class_train_nbc)

# Prediksi untuk data uji
y_pred_gpa_disc_nbc = model_gpa_disc_nbc_inf.predict(X_test_nbc_transformed)
y_pred_grade_class_nbc = model_grade_class_nbc_inf.predict(X_test_nbc_transformed)

# Evaluasi model
st.write("\nAkurasi GPA_Disc:", accuracy_score(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc))
st.text(classification_report(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc))

st.write("\nAkurasi GradeClass:", accuracy_score(y_grade_class_test_nbc, y_pred_grade_class_nbc))
st.text(classification_report(y_grade_class_test_nbc, y_pred_grade_class_nbc))

st.markdown("---")

plt.figure(figsize=(12,8))
sns.heatmap(data_nbc.corr()[['GPA_Disc']].sort_values(by='GPA_Disc', ascending=False), annot=True, cmap="coolwarm")
plt.title("Sensitivity Analysis: Korelasi Variabel terhadap GPA_Disc", fontsize=16)
st.pyplot(plt)

# Contoh eksperimen dengan berbagai prior
prior_types = ['uniform', 'empirical']
for prior in prior_types:
    model = CategoricalNB(alpha=1.0, fit_prior=True)
    model.fit(X_train_nbc_transformed, y_gpa_disc_train_nbc)
    y_pred = model.predict(X_test_nbc_transformed)
    accuracy = accuracy_score(y_gpa_disc_test_nbc, y_pred)
    st.write(f"Akurasi dengan prior {prior}: {accuracy:.4f}")


st.title("Evaluasi Model")
st.subheader("Mean Absolute Error (MAE) atau Root Mean Squared Error (RMSE)")
# Bagian evaluasi GPA (discretized)
mae_gpa = mean_absolute_error(true_gpa_disc, predictions_gpa_disc)
rmse_gpa = np.sqrt(mean_squared_error(true_gpa_disc, predictions_gpa_disc))

st.write(f"MAE untuk GPA_Disc: {mae_gpa:.2f}")
st.write(f"RMSE untuk GPA_Disc: {rmse_gpa:.2f}")

st.subheader("Akurasi, Presisi, Recall, AUC -> GradeClass")
# Bagian evaluasi GradeClass
accuracy_grade_class = accuracy_score(true_grade_class, predictions_grade_class)
precision_grade_class = precision_score(true_grade_class, predictions_grade_class, average='weighted')
recall_grade_class = recall_score(true_grade_class, predictions_grade_class, average='weighted')

# Prediksi probabilitas (tetap sama)
prob_grade_class = []
for i in range(len(test_data_bn)):
    evidence = test_data_bn.iloc[i].drop(['GPA_Disc', 'GradeClass'])
    grade_class_prob = inference.query(variables=['GradeClass'], evidence=evidence)
    prob_grade_class.append(grade_class_prob.values)

true_grade_class_bin = label_binarize(true_grade_class, classes=np.unique(true_grade_class))
auc_grade_class = roc_auc_score(true_grade_class_bin, np.array(prob_grade_class), average='macro', multi_class='ovr')

# Output Streamlit
st.write(f"Akurasi Prediksi GradeClass: {accuracy_grade_class * 100:.2f}%")
st.write(f"Presisi untuk GradeClass: {precision_grade_class * 100:.2f}%") 
st.write(f"Recall untuk GradeClass: {recall_grade_class * 100:.2f}%")
st.write(f"AUC untuk GradeClass: {auc_grade_class:.2f}")

st.subheader("Akurasi, Presisi, Recall, AUC -> GPA_Disc")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# --- 1. Evaluasi Akurasi, Precision, Recall, F1-Score untuk GPA_Disc --- #
st.header("Evaluasi Klasifikasi GPA_Disc")

# Calculate metrics
accuracy_gpa_disc = accuracy_score(true_gpa_disc, predictions_gpa_disc)
precision_gpa_disc = precision_score(true_gpa_disc, predictions_gpa_disc, average='weighted')
recall_gpa_disc = recall_score(true_gpa_disc, predictions_gpa_disc, average='weighted')
f1_gpa_disc = f1_score(true_gpa_disc, predictions_gpa_disc, average='weighted')

# Display metrics in columns
col1, col2 = st.columns(2)
with col1:
    st.metric("Akurasi", f"{accuracy_gpa_disc * 100:.2f}%")
    st.metric("Presisi", f"{precision_gpa_disc * 100:.2f}%")
with col2:
    st.metric("Recall", f"{recall_gpa_disc * 100:.2f}%")
    st.metric("F1-Score", f"{f1_gpa_disc * 100:.2f}%")

# --- 2. AUC untuk Multi-Class --- #
st.subheader("Evaluasi AUC untuk GPA_Disc")

# Calculate probabilities (keeping original calculation)
prob_gpa_disc = []
for i in range(len(test_data_bn)):
    evidence = test_data_bn.iloc[i].drop(['GPA_Disc', 'GradeClass'])
    gpa_disc_prob = inference.query(variables=['GPA_Disc'], evidence=evidence)
    prob_gpa_disc.append(gpa_disc_prob.values)

# Binarize labels and calculate AUC
true_gpa_disc_bin = label_binarize(true_gpa_disc, classes=np.unique(true_gpa_disc))
auc_gpa_disc = roc_auc_score(true_gpa_disc_bin, np.array(prob_gpa_disc), average='macro', multi_class='ovr')

# Display AUC with progress bar
st.metric("AUC Score", f"{auc_gpa_disc:.2f}")
st.progress(float(auc_gpa_disc), text=f"Kualitas Model: {'Baik' if auc_gpa_disc > 0.7 else 'Cukup'}")
