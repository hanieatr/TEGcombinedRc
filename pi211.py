import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os



# -------------------------------
# 1Custom model introduction
# -------------------------------
class CustomModel(tf.keras.Model):
    def __init__(self, input_shape, y_scaler, feature_scaler, A_rc_idx, A_cp_idx, categorical_indices):
        super(CustomModel, self).__init__()
        self.y_scaler = y_scaler
        self.feature_scaler = feature_scaler
        self.A_rc_idx = A_rc_idx
        self.A_cp_idx = A_cp_idx
        self.categorical_indices = categorical_indices

        self.shared_layers = [
            tf.keras.layers.Dense(128, activation='swish', kernel_constraint=tf.keras.constraints.max_norm(0.9)),
            tf.keras.layers.Dense(128, activation='swish', kernel_constraint=tf.keras.constraints.max_norm(0.9)),
            tf.keras.layers.Dense(128, activation='swish', kernel_constraint=tf.keras.constraints.max_norm(0.9)),
            tf.keras.layers.Dense(2)
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)
        return x

    def _apply_custom_scaling(self, X_transformed):
        """
        Apply the same scaling logic as in preprocessing:
        - Scale continuous features only
        - Keep categorical features as 0/1
        """
        continuous_indices = list(range(len(X_transformed[0]) - 5))

        # Split continuous and categorical features
        X_continuous = X_transformed[:, continuous_indices]
        X_categorical = X_transformed[:, self.categorical_indices]

        # Scale only continuous features
        X_continuous_scaled = self.feature_scaler.transform(X_continuous)

        # Concatenate scaled continuous with unscaled categorical
        X_scaled = np.concatenate([X_continuous_scaled, X_categorical], axis=1)

        return X_scaled

    def predict_from_raw(self, X_raw):
        """
        Predict from raw input values (without log transformation applied by user)
        X_raw should have the original A_rc and A_cp values
        """
        # Create a copy to avoid modifying the original
        X_transformed = X_raw.copy()

        # Apply log transformation to A_rc and A_cp columns
        X_transformed[:, self.A_rc_idx] = np.log(X_transformed[:, self.A_rc_idx])
        X_transformed[:, self.A_cp_idx] = np.log(X_transformed[:, self.A_cp_idx])

        # Apply custom scaling (continuous only, preserve categorical)
        X_scaled = self._apply_custom_scaling(X_transformed)

        # Get predictions in scaled space
        y_pred_scaled = super().predict(X_scaled, verbose=0)

        # Inverse transform to get actual predictions
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)

        return y_pred

    def predict(self, x, **kwargs):
        """Standard predict method for already transformed and scaled data"""
        y_pred_scaled = super().predict(x, **kwargs)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        return y_pred

# -------------------------------
@st.cache_resource
def load_models():
    # all-f
    all_features_columns = [
        'A_rc', 'A_cp', 'G', 'V', 'T_a', 'A', 'L', 'optical', 'alpha_cooler',
        'epsilon_1', 'epsilon_2', 'epsilon_3', 'epsilon_4',
        'taw_1', 'taw_2', 'taw_3', 'taw_4',
        'N', 'delta_cp', 'delta_cu', 'delta_rc', 'k_rc', 'delta_Al', 'k_Al',
        'BiTe', 'PbTe', 'Si80Ge20', 'SnSe', 'Bi0.5Sb1.5Te3'
    ]
    all_A_rc_idx = all_features_columns.index('A_rc')
    all_A_cp_idx = all_features_columns.index('A_cp')
    all_categorical_indices = list(range(len(all_features_columns) - 5, len(all_features_columns)))

    # f-s
    selected_features_columns = [
        'A_rc', 'A_cp', 'G', 'V', 'T_a', 'A', 'L', 'optical', 'alpha_cooler', 'taw_2',
        'BiTe', 'PbTe', 'Si80Ge20', 'SnSe', 'Bi0.5Sb1.5Te3'
    ]
    sel_A_rc_idx = selected_features_columns.index('A_rc')
    sel_A_cp_idx = selected_features_columns.index('A_cp')
    sel_categorical_indices = list(range(len(selected_features_columns) - 5, len(selected_features_columns)))

    # compelete iles
    all_weights = "Full_custom_model_weights.weights.h5"
    full_preparing_obj="Full_preprocessing_objects.pkl"

    # -selection files
    sel_weights = "FS_custom_model_weights.weights.h5"
    sel_preparing_obj = "preprocessing_objects.pkl"


    models = {}

    # help function
    def load_single_model(weights_path,preparing_obj_path, name):
        for path, fname in [(weights_path, "Weights"), (preparing_obj_path, "Preparing_objects=preparing_objects")]:
            if not os.path.exists(path):
                st.error(f"{fname} not found for **{name}**: `{path}`\nPlace it in the app folder.")
                st.stop()

        with open(preparing_obj_path, 'rb') as f:
            preprocessing_objects = pickle.load(f)

        model = CustomModel(
            input_shape=preprocessing_objects['input_shape'],
            y_scaler=preprocessing_objects['y_scaler'],
            feature_scaler=preprocessing_objects['feature_scaler'],
            A_rc_idx=preprocessing_objects['A_rc_idx'],
            A_cp_idx=preprocessing_objects['A_cp_idx'],
            categorical_indices=preprocessing_objects['categorical_indices']
        )

        dummy_input = np.zeros((1, preprocessing_objects['input_shape'][0]))
        _ = model(dummy_input, training=False)

        model.load_weights(weights_path)
        return model, preprocessing_objects

    # compelete model loading
    models['all'] = load_single_model(
        all_weights,   full_preparing_obj,
        "All Features Model"
    )

    # selection model loading
    models['selected'] = load_single_model(
        sel_weights, sel_preparing_obj,
        "Selected Features Model"
    )

    st.success("Both models loaded successfully!")
    return models


# بارگذاری مدل‌ها
models = load_models()
model_all, _, = models['all']
model_sel, _, = models['selected']

# -------------------------------
# 3. UI
# -------------------------------
st.set_page_config(page_title="Thermoelectric Predictor", layout="wide")

col1, col2 = st.columns(2)
if col1.button("Guide"):
    col1.subheader("How to use this app?")
    col1.image("gd.jpg", use_container_width=True)
    col1.markdown("**Enter parameters → Click Predict → Compare results!**", unsafe_allow_html=True)

if col2.button("About Us"):
    col2.subheader("About Us")
    col2.markdown("**Team:** A.Mousavi, H. Atrian Seresht, H. Aghakhani, B. Baghapour", unsafe_allow_html=True)

st.title("Thermoelectric Module Prediction")
########multiple pics
# -------------------------------
# گالری تصاویر (جایگزین st.image("gui.png") کن)
# -------------------------------
st.markdown("---")
st.subheader("pictures of different sections of system")

image_paths = [
    "gui.png",
    "gui_next1.png",
    "gui_next2.png" ]

# حالت اولیه: اولین عکس
if 'img_idx' not in st.session_state:
    st.session_state.img_idx = 0
n_images = len(image_paths)

# دکمه‌های قبلی و بعدی
col_prev, col_img, col_next = st.columns([1, 6, 1])

with col_prev:
    if st.button("Previous", use_container_width=True):
        st.session_state.img_idx = (st.session_state.img_idx - 1) % n_images

with col_img:
    current_img = image_paths[st.session_state.img_idx]
    st.image(current_img, use_container_width=True)

    # نمایش شماره عکس
    st.caption(f"picture {st.session_state.img_idx + 1} from {n_images}")

with col_next:
    if st.button("Next", use_container_width=True):
        st.session_state.img_idx = (st.session_state.img_idx + 1) % n_images

# -------------------------------
# Model Selection
# -------------------------------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose Prediction Method",
    [
        "Complete Model with All Features",
        "Feature Selection Model",
        "Equation based Model"
    ]
)

is_all_features = "Complete Model" in model_choice
is_feature_selection = "Feature Selection" in model_choice
is_eq = "Equation based Model" in model_choice


current_model = model_all if is_all_features else model_sel

st.sidebar.success(f"Selected: **{model_choice.split(' (')[0]}**")

# -------------------------------
# Common Inputs (همیشه)

materials = ["BiTe", "PbTe", "Si80Ge20", "SnSe", "Bi0.5Sb1.5Te3"]
material_choice = st.sidebar.radio("Material", materials)
material_encoding = [1 if material_choice == m else 0 for m in materials]
material_val = {"BiTe": 1.0, "PbTe": 2.0, "Si80Ge20": 3.0, "SnSe": 4.0, "Bi0.5Sb1.5Te3": 5.0}[material_choice]

A_rc = st.sidebar.number_input("A_rc (m²)", 1e-6, 0.1, 0.001, step=1e-6, format="%.6f")
A_cp = st.sidebar.number_input("A_cp (m²)", 1e-6, 0.1, 0.001, step=1e-6, format="%.6f")
G = st.sidebar.number_input("G (W/m²)", 10.0, 1500.0, 800.0, step=10.0)
V = st.sidebar.number_input("V (m/s)", 0.0, 10.0, 2.0, step=0.1)
T_a = st.sidebar.number_input("T_a (°k)", 0.0, 400.0, 25.0)
A = st.sidebar.number_input("Area A (m²)", 1e-9, 0.1, 1e-8, step=1e-7, format="%.9f")
L = st.sidebar.number_input("Length L (m)", 0.000001, 1.0, 0.00001, step=0.000001, format="%.6f")
optical = st.sidebar.number_input("Optical Efficiency", 0.01, 1.0, 0.9, step=0.01)
alpha_cooler = st.sidebar.number_input("α_cooler", 0.01, 1.0, 0.8, step=0.01)
taw_2 = st.sidebar.number_input("τ₂", 0.01, 1.0, 0.9, step=0.01)
CR = st.sidebar.number_input("CR", 0.1, 100.0, 10.0, step=0.1)
N = st.sidebar.number_input("N", 1, 300, 127, step=1)

# -------------------------------
# State-Specific Inputs
# -------------------------------
if is_all_features:
    epsilon_1 = st.sidebar.number_input("ε₁", 0.01, 1.0, 0.9, step=0.01)
    epsilon_2 = st.sidebar.number_input("ε₂", 0.01, 1.0, 0.9, step=0.01)
    epsilon_3 = st.sidebar.number_input("ε₃", 0.01, 1.0, 0.9, step=0.01)
    epsilon_4 = st.sidebar.number_input("ε₄", 0.01, 1.0, 0.9, step=0.01)
    taw_1 = st.sidebar.number_input("τ₁", 0.01, 1.0, 0.9, step=0.01)
    taw_3 = st.sidebar.number_input("τ₃", 0.01, 1.0, 0.9, step=0.01)
    taw_4 = st.sidebar.number_input("τ₄", 0.01, 1.0, 0.9, step=0.01)
    delta_cp = st.sidebar.number_input("Δ_cp (m)", 1e-6, 1.0, 1e-6, step=1e-6, format="%.6f")
    delta_cu = st.sidebar.number_input("Δ_cu (m)", 1e-6, 1.0, 1e-6, step=1e-6, format="%.6f")
    delta_rc = st.sidebar.number_input("Δ_rc (m)", 1e-6, 1.0, 1e-6, step=1e-6, format="%.6f")
    k_rc = st.sidebar.number_input("k_rc (W/m·K)", 0.1, 500.0, 200.0, step=1.0, format="%.6f")
    delta_Al = st.sidebar.number_input("Δ_Al (m)", 1e-6, 1.0, 1e-6, step=1e-6, format="%.6f")
    k_Al = st.sidebar.number_input("k_Al (W/m·K)", 0.1, 500.0, 237.0, step=1.0, format="%.6f")


elif is_eq:
    epsilon_2 = st.sidebar.number_input("ε₂ (T_h)", 0.01, 1.0, 0.9, step=0.01)
    taw_4 = st.sidebar.number_input("τ₄ (T_h)", 0.01, 1.0, 0.9, step=0.01)
    N = st.sidebar.number_input("N (T_h)", 1, 300, 127, step=1)
    epsilon_1 = st.sidebar.number_input("ε₁ (T_c)", 0.01, 1.0, 0.9, step=0.01)
    epsilon_3 = st.sidebar.number_input("ε₃ (T_c)", 0.01, 1.0, 0.9, step=0.01)
    epsilon_4 = st.sidebar.number_input("ε₄ (T_c)", 0.01, 1.0, 0.9, step=0.01)
    taw_1 = st.sidebar.number_input("τ₁ (T_c)", 0.01, 1.0, 0.9, step=0.01)


# -------------------------------
# ML Input Arrays
# -------------------------------
if is_all_features or is_feature_selection:
    optical_final = optical * CR * G
    if is_all_features:
        input_raw = np.array([[
            A_rc, A_cp, G, V, T_a, A, L, optical_final, alpha_cooler,
            epsilon_1, epsilon_2, epsilon_3, epsilon_4,
            taw_1, taw_2, taw_3, taw_4,
            N, delta_cp, delta_cu, delta_rc, k_rc, delta_Al, k_Al,
            *material_encoding
        ]], dtype=np.float32)
    else:
        input_raw = np.array([[
            A_rc, A_cp, G, V, T_a, A, L, optical_final, alpha_cooler, taw_2,
            *material_encoding
        ]], dtype=np.float32)

# -------------------------------
# Equation Functions
# -------------------------------
def calculate_th():
    A_rc_A_cp = A_rc / A_cp
    OpticalCRG = optical * CR * G
    Abs = np.abs
    sqrt = np.sqrt
    result_k=-1.6873049663104 * A * A_rc_A_cp * N * T_a * taw_2 / V + 0.0315112577259374 * G * alpha_cooler + 0.960638786367383 * OpticalCRG * (
                0.269938129308069 * sqrt(A_cp * sqrt(L) * sqrt(
            (A_cp + Abs(A_rc / (material_val ** 2 - 17.585363))) / (V - 0.6841718)) / A_rc) + 0.06287279 * A_cp / (
                            A * N * (T_a - material_val * (
                                -2.1007333 * A * G * N* V + 23.5187911711495)))) + 0.960638786367383 * T_a - 0.457287360736074 * V * taw_2 + 13.6620475672063 * (
                0.137695339793725 * epsilon_2 + 0.137695339793725 * taw_4 - 1) ** 2 + 1.9786514124226
    return result_k

def calculate_tc():
    OpticalCRG = optical * CR * G
    sqrt = np.sqrt
    result_k= 0.03566222*G*(-3.0763502*L*(material_val + (alpha_cooler + epsilon_1**(1.2088965/V) + epsilon_3)**2) + alpha_cooler + 0.0507079602572312) + OpticalCRG/(0.03968428*T_a*sqrt(A_rc*(6.9153543 + V/sqrt(A_cp))/A_cp) - 14.407003) + T_a + (epsilon_2*(-0.026225813*A_rc*T_a*epsilon_4/sqrt(A_cp*V) - 1.9661237) - 0.38108942*taw_1/epsilon_2)*(taw_2*(epsilon_1 + taw_2) + taw_4) - 0.16518472
    return result_k


def get_material_properties(material, T_h_k, T_c_k):
    T_mean = (T_h_k + T_c_k) / 2.0

    if material == "BiTe":
        alpha_p_Th = (22224 + 930.6 * T_h_k - 0.9905 * (T_h_k ** 2)) * 10 ** (-9)
        alpha_p_Tave = (22224 + 930.6 * T_mean - 0.9905 * (T_mean ** 2)) * 10 ** (-9)
        alpha_p_Tc = (22224 + 930.6 * T_c_k - 0.9905 * (T_c_k ** 2)) * 10 ** (-9)
        alpha_p = (alpha_p_Th + alpha_p_Tc + 2 * alpha_p_Tave) / 4

        alpha_n_Th = -(22224 + 930.6 * T_h_k - 0.9905 * (T_h_k ** 2)) * 10 ** (-9)
        alpha_n_Tave = -(22224 + 930.6 * T_mean - 0.9905 * (T_mean ** 2)) * 10 ** (-9)
        alpha_n_Tc = -(22224 + 930.6 * T_c_k - 0.9905 * (T_c_k ** 2)) * 10 ** (-9)
        alpha_n = (alpha_n_Th + alpha_n_Tc + 2 * alpha_n_Tave) / 4

        rho_p_Th = (5112 + 163.4 * T_h_k + 0.6279 * (T_h_k ** 2)) * 10 ** (-10)
        rho_p_Tave = (5112 + 163.4 * T_mean + 0.6279 * (T_mean ** 2)) * 10 ** (-10)
        rho_p_Tc = (5112 + 163.4 * T_c_k + 0.6279 * (T_c_k ** 2)) * 10 ** (-10)
        rho_p = (rho_p_Th + rho_p_Tc + 2 * rho_p_Tave) / 4

        rho_n_Th = (5112 + 163.4 * T_h_k + 0.6279 * (T_h_k ** 2)) * 10 ** (-10)
        rho_n_Tave = (5112 + 163.4 * T_mean + 0.6279 * (T_mean ** 2)) * 10 ** (-10)
        rho_n_Tc = (5112 + 163.4 * T_c_k + 0.6279 * (T_c_k ** 2)) * 10 ** (-10)
        rho_n = (rho_n_Th + rho_n_Tc + 2 * rho_n_Tave) / 4

        k_p_Th = (62605 - 277.7 * T_h_k + 0.4131 * (T_h_k ** 2)) * 10 ** (-4)
        k_p_Tave = (62605 - 277.7 * T_mean + 0.4131 * (T_mean ** 2)) * 10 ** (-4)
        k_p_Tc = (62605 - 277.7 * T_c_k + 0.4131 * (T_c_k ** 2)) * 10 ** (-4)
        k_p = (k_p_Th + k_p_Tc + 2 * k_p_Tave) / 4

        k_n_Th = (62605 - 277.7 * T_h_k + 0.4131 * (T_h_k ** 2)) * 10 ** (-4)
        k_n_Tave = (62605 - 277.7 * T_mean + 0.4131 * (T_mean ** 2)) * 10 ** (-4)
        k_n_Tc = (62605 - 277.7 * T_c_k + 0.4131 * (T_c_k ** 2)) * 10 ** (-4)
        k_n = (k_n_Th + k_n_Tc + 2 * k_n_Tave) / 4
        alpha = (alpha_p - alpha_n) * N
        return alpha, rho_p, rho_n, k_p, k_n

    elif material == "PbTe":
        alpha_p_Th = (-1.5799 * 10 ** (-4) + 7.6928 * 10 ** (-7) * T_h_k - 8.1391 * 10 ** (-11) * (T_h_k ** 2) - 3.1591 * 10 ** (
            -13) * (T_h_k ** 3))
        alpha_p_Tave = (-1.5799 * 10 ** (-4) + 7.6928 * 10 ** (-7) * T_mean - 8.1391 * 10 ** (-11) * (
                    T_mean ** 2) - 3.1591 * 10 ** (-13) * (T_mean ** 3))
        alpha_p_Tc = (-1.5799 * 10 ** (-4) + 7.6928 * 10 ** (-7) * T_c_k - 8.1391 * 10 ** (-11) * (T_c_k ** 2) - 3.1591 * 10 ** (
            -13) * (T_c_k ** 3))
        alpha_p = (alpha_p_Th + alpha_p_Tc + 2 * alpha_p_Tave) / 4

        alpha_n_Th = (-1.1317 * 10 ** (-4) + 4.5166 * 10 ** (-7) * T_h_k - 1.4263 * 10 ** (-9) * (T_h_k ** 2) + 8.8024 * 10 ** (
            -13) * (T_h_k ** 3))
        alpha_n_Tave = (-1.1317 * 10 ** (-4) + 4.5166 * 10 ** (-7) * T_mean - 1.4263 * 10 ** (-9) * (
                    T_mean ** 2) + 8.8024 * 10 ** (-13) * (T_mean ** 3))
        alpha_n_Tc = (-1.1317 * 10 ** (-4) + 4.5166 * 10 ** (-7) * T_c_k - 1.4263 * 10 ** (-9) * (T_c_k ** 2) + 8.8024 * 10 ** (
            -13) * (T_c_k ** 3))
        alpha_n = (alpha_n_Th + alpha_n_Tc + 2 * alpha_n_Tave) / 4


        rho_p_Th = (1.1182 * 10 ** (-5) - 8.5114 * 10 ** (-8) * T_h_k + 2.5123 * 10 ** (-10) * (T_h_k ** 2) - 1.5365 * 10 ** (
            -13) * (T_h_k ** 3))
        rho_p_Tave = (1.1182 * 10 ** (-5) - 8.5114 * 10 ** (-8) * T_mean + 2.5123 * 10 ** (-10) * (
                    T_mean ** 2) - 1.5365 * 10 ** (-13) * (T_mean ** 3))
        rho_p_Tc = (1.1182 * 10 ** (-5) - 8.5114 * 10 ** (-8) * T_c_k + 2.5123 * 10 ** (-10) * (T_c_k ** 2) - 1.5365 * 10 ** (
            -13) * (T_c_k ** 3))
        rho_p = (rho_p_Th + rho_p_Tc + 2 * rho_p_Tave) / 4

        rho_n_Th = (1.8911 * 10 ** (-5) - 7.6049 * 10 ** (-8) * T_h_k + 1.2018 * 10 ** (-10) * (T_h_k ** 2) - 1.9315 * 10 ** (
            -14) * (T_h_k ** 3))
        rho_n_Tave = (1.8911 * 10 ** (-5) - 7.6049 * 10 ** (-8) * T_mean + 1.2018 * 10 ** (-10) * (
                    T_mean ** 2) - 1.9315 * 10 ** (-14) * (T_mean ** 3))
        rho_n_Tc = (1.8911 * 10 ** (-5) - 7.6049 * 10 ** (-8) * T_c_k + 1.2018 * 10 ** (-10) * (T_c_k ** 2) - 1.9315 * 10 ** (
            -14) * (T_c_k ** 3))
        rho_n = (rho_n_Th + rho_n_Tc + 2 * rho_n_Tave) / 4


        k_p_Th = 9.3803 - 2.7721 * 10 ** (-2) * T_h_k + 3.0486 * 10 ** (-5) * (T_h_k ** 2) - 1.0899 * 10 ** (-8) * (T_h_k ** 3)
        k_p_Tave = 9.3803 - 2.7721 * 10 ** (-2) * T_mean + 3.0486 * 10 ** (-5) * (T_mean ** 2) - 1.0899 * 10 ** (-8) * (
                    T_mean ** 3)
        k_p_Tc = 9.3803 - 2.7721 * 10 ** (-2) * T_c_k + 3.0486 * 10 ** (-5) * (T_c_k ** 2) - 1.0899 * 10 ** (-8) * (T_c_k ** 3)
        k_p = (k_p_Th + k_p_Tc + 2 * k_p_Tave) / 4

        k_n_Th = 8.7124 - 2.5897 * 10 ** (-2) * T_h_k + 2.9196 * 10 **(-5) * (T_h_k ** 2) - 1.0776 * 10 ** (-8) * (T_h_k ** 3)
        k_n_Tave = 8.7124 - 2.5897 * 10 ** (-2) * T_mean + 2.9196 * 10 ** (-5) * (T_mean ** 2) - 1.0776 * 10 ** (-8) * (
                    T_mean ** 3)
        k_n_Tc = 8.7124 - 2.5897 * 10 ** (-2) * T_c_k + 2.9196 * 10 ** (-5) * (T_c_k ** 2) - 1.0776 * 10 ** (-8) * (T_c_k ** 3)
        k_n = (k_n_Th + k_n_Tc + 2 * k_n_Tave) / 4
        alpha = N * (alpha_p - alpha_n)
        return alpha, rho_p, rho_n, k_p, k_n

    elif material == "Si80Ge20":
        alpha_p_Th = (3.36296 * 10 ** (-6) + 0.45411 * 10 ** (-6) * T_h_k - 4.77955 * 10 ** (-10) * (
                    T_h_k ** 2) + 2.39307 * 10 ** (-13) * (T_h_k ** 3))
        alpha_p_Tave = (3.36296 * 10 ** (-6) + 0.45411 * 10 ** (-6) * T_mean - 4.77955 * 10 ** (-10) * (
                    T_mean ** 2) + 2.39307 * 10 ** (-13) * (T_mean ** 3))
        alpha_p_Tc = (3.36296 * 10 ** (-6) + 0.45411 * 10 ** (-6) * T_c_k - 4.77955 * 10 ** (-10) * (
                    T_c_k ** 2) + 2.39307 * 10 ** (-13) * (T_c_k ** 3))
        alpha_p = (alpha_p_Th + alpha_p_Tc + 2 * alpha_p_Tave) / 4

        alpha_n_Th = (-1.0114 * 10 ** (-4) + 5.41787 * 10 ** (-9) * T_h_k - 6.13998 * 10 ** (-10) * (
                    T_h_k ** 2) + 4.1635 * 10 ** (-13) * (T_h_k ** 3))
        alpha_n_Tave = (-1.0114 * 10 ** (-4) + 5.41787 * 10 ** (-9) * T_mean - 6.13998 * 10 ** (-10) * (
                    T_mean ** 2) + 4.1635 * 10 ** (-13) * (T_mean ** 3))
        alpha_n_Tc = (-1.0114 * 10 ** (-4) + 5.41787 * 10 ** (-9) * T_c_k - 6.13998 * 10 ** (-10) * (
                    T_c_k ** 2) + 4.1635 * 10 ** (-13) * (T_c_k ** 3))
        alpha_n = (alpha_n_Th + alpha_n_Tc + 2 * alpha_n_Tave) / 4

        rho_p_Th = 1 / (148681.26 - 265.812 * T_h_k + 0.28773 * (T_h_k ** 2) - 1.27738 * 10 ** (-4) * (T_h_k ** 3))
        rho_p_Tave = 1 / (148681.26 - 265.812 * T_mean + 0.28773 * (T_mean ** 2) - 1.27738 * 10 ** (-4) * (T_mean ** 3))
        rho_p_Tc = 1 / (148681.26 - 265.812 * T_c_k + 0.28773 * (T_c_k ** 2) - 1.27738 * 10 ** (-4) * (T_c_k ** 3))
        rho_p = (rho_p_Th + rho_p_Tc + 2 * rho_p_Tave) / 4

        rho_n_Th = 1 / (8868.71 + 96.0086 * T_h_k - 0.193 * (T_h_k ** 2) + 1.05692 * 10 ** (-4) * (T_h_k ** 3))
        rho_n_Tave = 1 / (8868.71 + 96.0086 * T_mean - 0.193 * (T_mean ** 2) + 1.05692 * 10 ** (-4) * (T_mean ** 3))
        rho_n_Tc = 1 / (8868.71 + 96.0086 * T_c_k - 0.193 * (T_c_k ** 2) + 1.05692 * 10 ** (-4) * (T_c_k ** 3))
        rho_n = (rho_n_Th + rho_n_Tc + 2 * rho_n_Tave) / 4

        k_p_Th = 1.81114 + 0.002 * T_h_k - 1.85862 * 10 ** (-6) * (T_h_k ** 2) + 5.10263 * 10 ** (-10) * (T_h_k ** 3)
        k_p_Tave = 1.81114 + 0.002 * T_mean - 1.85862 * 10 ** (-6) * (T_mean ** 2) + 5.10263 * 10 ** (-10) * (
                    T_mean ** 3)
        k_p_Tc = 1.81114 + 0.002 * T_c_k - 1.85862 * 10 ** (-6) * (T_c_k ** 2) + 5.10263 * 10 ** (-10) * (T_c_k ** 3)
        k_p = (k_p_Th + k_p_Tc + 2 * k_p_Tave) / 4

        k_n_Th = 0.88589 - 6.40814 * 10 ** (-4) * T_h_k + 7.56926 * 10 ** (-7) * (T_h_k ** 2) - 9.06357 * 10 ** (
            -11) * (T_h_k ** 3)
        k_n_Tave = 0.88589 - 6.40814 * 10 ** (-4) * T_mean + 7.56926 * 10 ** (-7) * (T_mean ** 2) - 9.06357 * 10 ** (
            -11) * (T_mean ** 3)
        k_n_Tc = 0.88589 - 6.40814 * 10 ** (-4) * T_c_k + 7.56926 * 10 ** (-7) * (T_c_k ** 2) - 9.06357 * 10 ** (
            -11) * (T_c_k ** 3)
        k_n = (k_n_Th + k_n_Tc + 2 * k_n_Tave) / 4

        alpha = (alpha_p - alpha_n) *N
        return alpha, rho_p, rho_n, k_p, k_n


    elif material == "SnSe":
        alpha_p_Th = (
                    0.00218 - 1.3596 * 10 ** (-5) * T_h_k + 3.903 * 10 ** (-8) * (T_h_k ** 2) - 4.601 * 10 ** (-11) * (
                        T_h_k ** 3) + 1.877 * 10 ** (-14) * (T_h_k ** 4))
        alpha_p_Tave = (0.00218 - 1.3596 * 10 ** (-5) * T_mean + 3.903 * 10 ** (-8) * (T_mean ** 2) - 4.601 * 10 ** (
            -11) * (T_mean ** 3) + 1.877 * 10 ** (-14) * (T_mean ** 4))
        alpha_p_Tc = (
                    0.00218 - 1.3596 * 10 ** (-5) * T_c_k + 3.903 * 10 ** (-8) * (T_c_k ** 2) - 4.601 * 10 ** (-11) * (
                        T_c_k ** 3) + 1.877 * 10 ** (-14) * (T_c_k ** 4))
        alpha_p = (alpha_p_Th + alpha_p_Tc + 2 * alpha_p_Tave) / 4

        alpha_n_Th = (-0.00116 + 3.93837 * 10 ** (-6) * T_h_k - 7.88668 * 10 ** (-9) * (T_h_k ** 2) + 5.31009 * 10 ** (
            -12) * (T_h_k ** 3))
        alpha_n_Tave = (
                    -0.00116 + 3.93837 * 10 ** (-6) * T_mean - 7.88668 * 10 ** (-9) * (T_mean ** 2) + 5.31009 * 10 ** (
                -12) * (T_mean ** 3))
        alpha_n_Tc = (-0.00116 + 3.93837 * 10 ** (-6) * T_c_k - 7.88668 * 10 ** (-9) * (T_c_k ** 2) + 5.31009 * 10 ** (
            -12) * (T_c_k ** 3))
        alpha_n = (alpha_n_Th + alpha_n_Tc + 2 * alpha_n_Tave) / 4

        rho_p_Th = 1 / (1.3821 * 10 ** 5 - 1396.8 * T_h_k + 5.5985 * (T_h_k ** 2) - 1.1163 * 10 ** (-2) * (
                    T_h_k ** 3) + 1.1401 * 10 ** (-5) * (T_h_k ** 4) - 5.4087 * 10 ** (-9) * (
                                    T_h_k ** 5) + 8.3559 * 10 ** (-13) * (T_h_k ** 6))
        rho_p_Tave = 1 / (1.3821 * 10 ** 5 - 1396.8 * T_mean + 5.5985 * (T_mean ** 2) - 1.1163 * 10 ** (-2) * (
                    T_mean ** 3) + 1.1401 * 10 ** (-5) * (T_mean ** 4) - 5.4087 * 10 ** (-9) * (
                                      T_mean ** 5) + 8.3559 * 10 ** (-13) * (T_mean ** 6))
        rho_p_Tc = 1 / (1.3821 * 10 ** 5 - 1396.8 * T_c_k + 5.5985 * (T_c_k ** 2) - 1.1163 * 10 ** (-2) * (
                    T_c_k ** 3) + 1.1401 * 10 ** (-5) * (T_c_k ** 4) - 5.4087 * 10 ** (-9) * (
                                    T_c_k ** 5) + 8.3559 * 10 ** (-13) * (T_c_k ** 6))
        rho_p = (rho_p_Th + rho_p_Tc + 2 * rho_p_Tave) / 4

        rho_n_Th = 1 / (24188.25533 - 163.42861 * T_h_k + 0.3514 * (T_h_k ** 2) - 2.23021 * 10 ** (-4) * (T_h_k ** 3))
        rho_n_Tave = 1 / (
                    24188.25533 - 163.42861 * T_mean + 0.3514 * (T_mean ** 2) - 2.23021 * 10 ** (-4) * (T_mean ** 3))
        rho_n_Tc = 1 / (24188.25533 - 163.42861 * T_c_k + 0.3514 * (T_c_k ** 2) - 2.23021 * 10 ** (-4) * (T_c_k ** 3))
        rho_n = (rho_n_Th + rho_n_Tc + 2 * rho_n_Tave) / 4

        k_p_Th = 0.1329 + 0.00585 * T_h_k - 1.8947 * 10 ** (-5) * (T_h_k ** 2) + 2.1544 * 10 ** (-8) * (
                    T_h_k ** 3) - 8.2507 * 10 ** (-12) * (T_h_k ** 4)
        k_p_Tave = 0.1329 + 0.00585 * T_mean - 1.8947 * 10 ** (-5) * (T_mean ** 2) + 2.1544 * 10 ** (-8) * (
                    T_mean ** 3) - 8.2507 * 10 ** (-12) * (T_mean ** 4)
        k_p_Tc = 0.1329 + 0.00585 * T_c_k - 1.8947 * 10 ** (-5) * (T_c_k ** 2) + 2.1544 * 10 ** (-8) * (
                    T_c_k ** 3) - 8.2507 * 10 ** (-12) * (T_c_k ** 4)
        k_p = (k_p_Th + k_p_Tc + 2 * k_p_Tave) / 4

        k_n_Th = 3.43674 - 0.01476 * T_h_k + 2.55645 * 10 ** (-5) * (T_h_k ** 2) - 1.52631 * 10 ** (-8) * (T_h_k ** 3)
        k_n_Tave = 3.43674 - 0.01476 * T_mean + 2.55645 * 10 ** (-5) * (T_mean ** 2) - 1.52631 * 10 ** (-8) * (
                    T_mean ** 3)
        k_n_Tc = 3.43674 - 0.01476 * T_c_k + 2.55645 * 10 ** (-5) * (T_c_k ** 2) - 1.52631 * 10 ** (-8) * (T_c_k ** 3)
        k_n = (k_n_Th + k_n_Tc + 2 * k_n_Tave) / 4

        alpha = (alpha_p - alpha_n)*N
        return alpha, rho_p, rho_n, k_p, k_n


    elif material == "Bi0.5Sb1.5Te3":
        alpha_p_Th = (-525.98585 * 10 ** (-6) + 4.97271 * 10 ** (-6) * T_h_k - 0.00958 * 10 ** (-6) * (
                    T_h_k ** 2) + 4.8283 * 10 ** (-12) * (T_h_k ** 3))
        alpha_p_Tave = (-525.98585 * 10 ** (-6) + 4.97271 * 10 ** (-6) * T_mean - 0.00958 * 10 ** (-6) * (
                    T_mean ** 2) + 4.8283 * 10 ** (-12) * (T_mean ** 3))
        alpha_p_Tc = (-525.98585 * 10 ** (-6) + 4.97271 * 10 ** (-6) * T_c_k - 0.00958 * 10 ** (-6) * (
                    T_c_k ** 2) + 4.8283 * 10 ** (-12) * (T_c_k ** 3))
        alpha_p = (alpha_p_Th + alpha_p_Tc + 2 * alpha_p_Tave) / 4

        alpha_n_Th = (417.48349 * 10 ** (-6) - 3.94411 * 10 ** (-6) * T_h_k + 0.00689 * 10 ** (-6) * (
                    T_h_k ** 2) - 2.56705 * 10 ** (-12) * (T_h_k ** 3))
        alpha_n_Tave = (417.48349 * 10 ** (-6) - 3.94411 * 10 ** (-6) * T_mean + 0.00689 * 10 ** (-6) * (
                    T_mean ** 2) - 2.56705 * 10 ** (-12) * (T_mean ** 3))
        alpha_n_Tc = (417.48349 * 10 ** (-6) - 3.94411 * 10 ** (-6) * T_c_k + 0.00689 * 10 ** (-6) * (
                    T_c_k ** 2) - 2.56705 * 10 ** (-12) * (T_c_k ** 3))
        alpha_n = (alpha_n_Th + alpha_n_Tc + 2 * alpha_n_Tave) / 4

        rho_p_Th = 1 / (414595.81 - 2047.6 * T_h_k + 3.664 * (T_h_k ** 2) - 2.14407 * 10 ** (-3) * (T_h_k ** 3))
        rho_p_Tave = 1 / (414595.81 - 2047.6 * T_mean + 3.664 * (T_mean ** 2) - 2.14407 * 10 ** (-3) * (T_mean ** 3))
        rho_p_Tc = 1 / (414595.81 - 2047.6 * T_c_k + 3.664 * (T_c_k ** 2) - 2.14407 * 10 ** (-3) * (T_c_k ** 3))
        rho_p = (rho_p_Th + rho_p_Tc + 2 * rho_p_Tave) / 4

        rho_n_Th = 1 / (565547.725 - 2896.099 * T_h_k + 5.496 * (T_h_k ** 2) - 3.60488 * 10 ** (-3) * (T_h_k ** 3))
        rho_n_Tave = 1 / (565547.725 - 2896.099 * T_mean + 5.496 * (T_mean ** 2) - 3.60488 * 10 ** (-3) * (T_mean ** 3))
        rho_n_Tc = 1 / (565547.725 - 2896.099 * T_c_k + 5.496 * (T_c_k ** 2) - 3.60488 * 10 ** (-3) * (T_c_k ** 3))
        rho_n = (rho_n_Th + rho_n_Tc + 2 * rho_n_Tave) / 4

        k_p_Th = 5.03763 - 0.03172 * T_h_k + 7.23229 * 10 ** (-5) * (T_h_k ** 2) - 4.95402 * 10 ** (-8) * (T_h_k ** 3)
        k_p_Tave = 5.03763 - 0.03172 * T_mean + 7.23229 * 10 ** (-5) * (T_mean ** 2) - 4.95402 * 10 ** (-8) * (
                    T_mean ** 3)
        k_p_Tc = 5.03763 - 0.03172 * T_c_k + 7.23229 * 10 ** (-5) * (T_c_k ** 2) - 4.95402 * 10 ** (-8) * (T_c_k ** 3)
        k_p = (k_p_Th + k_p_Tc + 2 * k_p_Tave) / 4

        k_n_Th = 14.11288 - 0.09569 * T_h_k + 2.29075 * 10 ** (-4) * (T_h_k ** 2) - 1.72292 * 10 ** (-7) * (T_h_k ** 3)
        k_n_Tave = 14.11288 - 0.09569 * T_mean + 2.29075 * 10 ** (-4) * (T_mean ** 2) - 1.72292 * 10 ** (-7) * (
                    T_mean ** 3)
        k_n_Tc = 14.11288 - 0.09569 * T_c_k + 2.29075 * 10 ** (-4) * (T_c_k ** 2) - 1.72292 * 10 ** (-7) * (T_c_k ** 3)
        k_n = (k_n_Th + k_n_Tc + 2 * k_n_Tave) / 4

        alpha = (alpha_p - alpha_n)*N
        return alpha, rho_p, rho_n, k_p, k_n


def calculate_performance(T_h_k, T_c_k):
    delta_T = T_h_k - T_c_k

    # شرط مهم: اگر اختلاف دما کمتر از 1 کلوین باشه → عملاً توان صفره
    if delta_T <= 0.1:
        return {"I": 0.0, "V_out": 0.0, "Power": 0.0, "eta": 0.0, "Q_h": 0.0, "Q_c": 0.0}

    # اگر A, L, N صفر یا منفی باشن
    if A <= 0 or L <= 0 or N <= 0:
        return {"I": 0.0, "V_out": 0.0, "Power": 0.0}
    alpha, rho_p, rho_n, k_p, k_n = get_material_properties(material_choice, T_h_k, T_c_k)

    R_in = N * (rho_p + rho_n) * L / A
    R_leg = L / (N * (k_p + k_n) * A)
    R_load = R_in
    I = (alpha * delta_T) / (2 * R_in)   # این فرمول دقیق‌تره!
    V_out = I * R_load
    Power = I * V_out
    # گرمای جذب و دفع (برای بازده)
    Q_h = alpha * I * T_h_k - 0.5 * I**2 * R_in + delta_T / R_leg
    Q_c = alpha * I * T_c_k + 0.5 * I**2 * R_in + delta_T / R_leg
    return {
        "I": max(I, 0),
        "V_out": max(V_out, 0),
        "Power": max(Power, 0),
        "delta_T": delta_T
    }

# -------------------------------
# Prediction
# -------------------------------
if st.button("Prediction ", type="primary"):
    with st.spinner("Calculating..."):
        results = {}

        if is_all_features or is_feature_selection:
            y_pred = current_model.predict_from_raw(input_raw)
            T_h_ml, T_c_ml = y_pred[0]
            T_h_ml -= 273.15
            T_c_ml -= 273.15
            name = "All Features" if is_all_features else "Feature Selection"
            results[name] = (T_h_ml, T_c_ml)

            T_h_k = T_h_ml + 273.15
            T_c_k = T_c_ml + 273.15

            perf = calculate_performance(T_h_k, T_c_k)

            st.success("successful prediction")

            st.markdown("---")
            st.subheader(f" Thermoelectric performance-material: **{material_choice}**")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("current (I)", f"{perf['I']:.5f} A")
                st.metric("voltage (V)", f"{perf['V_out']:.5f} V")
            with c2:
                st.metric("power density (P)", f"{perf['Power']:.6f} W")



        if is_eq:
            T_h_eq = calculate_th() - 273.15  # کلوین → °C
            T_c_eq = calculate_tc() - 273.15  # کلوین → °C
            results["T_h Equation"] = (T_h_eq, None)
            results["T_c Equation"] = (None, T_c_eq)
        st.success("Done!")
        if is_eq:
            with st.expander("Equations"):
                if is_eq:
                    st.latex(r"""
                    \begin{aligned}
                    T_h &= -1.6873 \cdot \frac{A \cdot A_{rc}}{A_{cp}} \cdot N \cdot T_a \cdot \frac{\tau_2}{V} \\
                    &\quad + 0.0315 \cdot G \cdot \alpha_{cooler} \\
                    &\quad + 0.9606 \cdot \eta \cdot CR \cdot G \cdot \left( 
                        \frac{0.2699 \cdot \sqrt{A_{cp} \sqrt{L} \sqrt{\frac{A_{cp} + \left| \frac{A_{rc}}{m^2 - 17.585} \right|}{V - 0.684}}}}{A_{rc}} 
                    \right) \\
                    &\quad + \frac{0.0629 \cdot A_{cp}}{A \cdot N \cdot \left( T_a - m \cdot (-2.1007 \cdot A \cdot G \cdot N \cdot V + 23.519) \right)} \\
                    &\quad + 0.9606 \cdot T_a - 0.4573 \cdot V \cdot \tau_2 \\
                    &\quad + 13.662 \cdot (0.1377 \cdot \varepsilon_2 + 0.1377 \cdot \tau_4 - 1)^2 + 1.979
                    \end{aligned}
                    """)

                    st.latex(r"""
                    \begin{aligned}
                    T_c &= 0.0357 \cdot G \cdot \left( -3.076 \cdot L \cdot (m + (\alpha_{cooler} + \varepsilon_1^{1.209/V} + \varepsilon_3)^2) + \alpha_{cooler} + 0.0507 \right) \\
                    &\quad + \frac{\eta \cdot CR \cdot G}{0.0397 \cdot T_a \cdot \sqrt{A_{rc} \cdot (6.915 + V/\sqrt{A_{cp}})/A_{cp}} - 14.407} \\
                    &\quad + T_a \\
                    &\quad + \left[ \varepsilon_2 \left( -0.0262 \cdot \frac{A_{rc} T_a \varepsilon_4}{\sqrt{A_{cp} V}} - 1.966 \right) - \frac{0.381 \cdot \tau_1}{\varepsilon_2} \right] \\
                    &\quad \cdot (\tau_2 (\varepsilon_1 + \tau_2) + \tau_4) - 0.165
                    \end{aligned}
                    """)

        cols = st.columns(len(results))
        for i, (name, (th, tc)) in enumerate(results.items()):
            with cols[i]:
                st.subheader(f"**{name}**")
                if th is not None: st.metric("T_h", f"{th:.3f} °C")
                if tc is not None: st.metric("T_c", f"{tc:.3f} °C")
                with st.expander("Kelvin"):
                    if th is not None: st.write(f"T_h: {th+273:.3f} K")
                    if tc is not None: st.write(f"T_c: {tc+273:.3f} K")