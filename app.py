import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import fetch_openml
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark industrial theme */
.stApp {
    background-color: #0e0e0e;
    color: #f0ece3;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.03em;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0ece3;
    border-bottom: 3px solid #e8c547;
    padding-bottom: 0.4rem;
    margin-bottom: 0.2rem;
    letter-spacing: -0.04em;
}

.subtitle {
    color: #888;
    font-size: 0.95rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 2rem;
}

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 4px solid #e8c547;
    padding: 1.2rem 1.5rem;
    border-radius: 4px;
    margin: 0.4rem 0;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e8c547;
}

.metric-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.prediction-box {
    background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
    border: 2px solid #e8c547;
    border-radius: 6px;
    padding: 2rem;
    text-align: center;
}

.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8c547;
}

.pred-confidence {
    font-size: 1rem;
    color: #aaa;
    margin-top: 0.3rem;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #e8c547;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.stSelectbox label, .stSlider label, .stRadio label {
    color: #ccc !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stButton > button {
    background: #e8c547 !important;
    color: #0e0e0e !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.05em;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #f5d660 !important;
    transform: translateY(-1px);
}

.tag {
    display: inline-block;
    background: #2a2a2a;
    color: #e8c547;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin: 0.2rem;
    letter-spacing: 0.05em;
}

.info-box {
    background: #151515;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #999;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "👕 T-shirt/Top", "👖 Trouser", "🧥 Pullover", "👗 Dress", "🧥 Coat",
    "👡 Sandal", "👔 Shirt", "👟 Sneaker", "👜 Bag", "👢 Ankle Boot"
]
CLASS_NAMES_SHORT = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    dataset = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    X, y = dataset.data, dataset.target.astype(int)
    X = X / 255.0
    return X, y

# ── Training ──────────────────────────────────────────────────────────────────
def build_hidden_layers(n_layers, neurons_per_layer):
    return tuple([neurons_per_layer] * n_layers)

def train_model(X_train, y_train, hidden_layers, activation, solver, max_iter, alpha):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        alpha=alpha,
        random_state=42,
        verbose=False,
        early_stopping=True,
        n_iter_no_change=10,
    )
    model.fit(X_scaled, y_train)
    return model, scaler

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Arquitectura de Red</div>', unsafe_allow_html=True)

    net_type = st.radio(
        "Tipo de Red",
        ["MLP (Shallow)", "DNN (Deep)"],
        help="MLP: 1-2 capas ocultas | DNN: 3+ capas ocultas"
    )

    if net_type == "MLP (Shallow)":
        n_layers = st.slider("Número de capas ocultas", 1, 2, 1)
        neurons = st.select_slider("Neuronas por capa", options=[32, 64, 128, 256], value=128)
    else:
        n_layers = st.slider("Número de capas ocultas", 3, 6, 3)
        neurons = st.select_slider("Neuronas por capa", options=[64, 128, 256, 512], value=256)

    st.markdown('<div class="section-header">🔧 Hiperparámetros</div>', unsafe_allow_html=True)

    activation = st.selectbox(
        "Función de Activación",
        ["relu", "tanh", "logistic", "identity"],
        index=0
    )

    solver = st.selectbox(
        "Optimizador",
        ["adam", "sgd", "lbfgs"],
        index=0
    )

    max_iter = st.slider("Iteraciones máximas", 50, 300, 100, 25)
    alpha = st.select_slider(
        "Regularización L2 (alpha)",
        options=[0.0001, 0.001, 0.01, 0.1],
        value=0.0001
    )

    st.markdown('<div class="section-header">📊 Datos de Entrenamiento</div>', unsafe_allow_html=True)
    train_size = st.slider("Muestras de entrenamiento", 5000, 30000, 10000, 5000)
    test_size = st.slider("Muestras de prueba", 1000, 5000, 2000, 500)

    train_button = st.button("🚀 Entrenar Modelo", use_container_width=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">FASHION MNIST</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">// clasificador de prendas · MLP / DNN · sklearn</div>', unsafe_allow_html=True)

# Architecture summary tags
hidden_layers = build_hidden_layers(n_layers, neurons)
arch_str = " → ".join([f"784"] + [str(neurons)] * n_layers + ["10"])
st.markdown(
    f'<span class="tag">{net_type}</span>'
    f'<span class="tag">Capas: {n_layers}</span>'
    f'<span class="tag">Neuronas: {neurons}</span>'
    f'<span class="tag">Activación: {activation}</span>'
    f'<span class="tag">Optimizador: {solver}</span>',
    unsafe_allow_html=True
)
st.markdown(f'<div style="font-family:Space Mono;font-size:0.75rem;color:#555;margin-top:0.5rem;margin-bottom:1.5rem;">{arch_str}</div>', unsafe_allow_html=True)

# ── Load data once ────────────────────────────────────────────────────────────
with st.spinner("Cargando Fashion MNIST..."):
    X, y = load_data()

# ── Training flow ─────────────────────────────────────────────────────────────
if train_button:
    idx = np.random.permutation(len(X))
    X_train, y_train = X[idx[:train_size]], y[idx[:train_size]]
    X_test, y_test = X[idx[train_size:train_size + test_size]], y[idx[train_size:train_size + test_size]]

    with st.spinner("Entrenando red neuronal..."):
        t0 = time.time()
        model, scaler = train_model(X_train, y_train, hidden_layers, activation, solver, max_iter, alpha)
        elapsed = time.time() - t0

    # Store in session
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["elapsed"] = elapsed
    st.session_state["X_all"] = X
    st.session_state["y_all"] = y

# ── Results ───────────────────────────────────────────────────────────────────
if "model" in st.session_state:
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    elapsed = st.session_state["elapsed"]

    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    n_iter = model.n_iter_

    # ── Performance metrics ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Desempeño del Modelo</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_iter}</div><div class="metric-label">Iteraciones</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.1f}s</div><div class="metric-label">Tiempo entreno</div></div>', unsafe_allow_html=True)
    with c4:
        total_params = 784 * neurons + sum([neurons * neurons] * (n_layers - 1)) + neurons * 10
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_params:,}</div><div class="metric-label">Parámetros</div></div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🎯 Predicción", "📊 Curva de Pérdida", "🔢 Matriz Confusión"])

    with tab1:
        st.markdown('<div class="section-header">Selecciona una imagen para clasificar</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            X_all = st.session_state["X_all"]
            y_all = st.session_state["y_all"]

            filter_class = st.selectbox(
                "Filtrar por clase",
                ["Todas"] + CLASS_NAMES_SHORT
            )

            if filter_class == "Todas":
                available_idx = np.arange(len(X_all))
            else:
                cls_id = CLASS_NAMES_SHORT.index(filter_class)
                available_idx = np.where(y_all == cls_id)[0]

            sample_idx = st.slider(
                "Índice de imagen",
                0, min(len(available_idx) - 1, 999), 0
            )

            real_idx = available_idx[sample_idx]
            img = X_all[real_idx].reshape(28, 28)
            true_label = y_all[real_idx]

            fig_img, ax = plt.subplots(figsize=(3, 3))
            fig_img.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            ax.imshow(img, cmap='inferno', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f"Real: {CLASS_NAMES_SHORT[true_label]}", 
                        color='#e8c547', fontsize=9, pad=8, fontfamily='monospace')
            st.pyplot(fig_img, use_container_width=False)
            plt.close()

            if st.button("⚡ Clasificar Imagen", use_container_width=True):
                st.session_state["predict_idx"] = real_idx

        with col_right:
            if "predict_idx" in st.session_state:
                p_idx = st.session_state["predict_idx"]
                x_sample = X_all[p_idx].reshape(1, -1)
                x_scaled = scaler.transform(x_sample)
                pred_class = model.predict(x_scaled)[0]
                proba = model.predict_proba(x_scaled)[0]

                correct = pred_class == y_all[p_idx]
                border_color = "#4caf50" if correct else "#f44336"
                icon = "✅" if correct else "❌"

                st.markdown(f"""
                <div class="prediction-box" style="border-color:{border_color}">
                    <div style="font-size:3rem">{CLASS_NAMES[pred_class].split()[0]}</div>
                    <div class="pred-label">{CLASS_NAMES_SHORT[pred_class]}</div>
                    <div class="pred-confidence">Confianza: {proba[pred_class]:.1%} {icon}</div>
                    <div style="font-size:0.8rem;color:#666;margin-top:0.5rem;font-family:monospace">
                        Real: {CLASS_NAMES_SHORT[y_all[p_idx]]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top-3 bar chart
                st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
                top3_idx = np.argsort(proba)[::-1][:5]
                fig_bar, ax2 = plt.subplots(figsize=(5, 2.5))
                fig_bar.patch.set_facecolor('#1a1a1a')
                ax2.set_facecolor('#1a1a1a')
                colors = ['#e8c547' if i == pred_class else '#2a2a2a' for i in top3_idx]
                bars = ax2.barh(
                    [CLASS_NAMES_SHORT[i] for i in top3_idx[::-1]],
                    [proba[i] for i in top3_idx[::-1]],
                    color=colors[::-1], edgecolor='none', height=0.6
                )
                ax2.set_xlim(0, 1)
                ax2.set_xlabel("Probabilidad", color='#888', fontsize=8)
                ax2.tick_params(colors='#999', labelsize=8)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#2a2a2a')
                ax2.xaxis.label.set_color('#888')
                st.pyplot(fig_bar, use_container_width=True)
                plt.close()
            else:
                st.markdown("""
                <div class="info-box" style="margin-top:3rem;text-align:center;">
                    <div style="font-size:2rem">👈</div>
                    <div>Selecciona una imagen y presiona<br><strong>Clasificar Imagen</strong></div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Curva de pérdida durante el entrenamiento</div>', unsafe_allow_html=True)

        if hasattr(model, 'loss_curve_'):
            fig_loss, ax = plt.subplots(figsize=(9, 4))
            fig_loss.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#141414')
            ax.plot(model.loss_curve_, color='#e8c547', linewidth=2, label='Training loss')
            if hasattr(model, 'validation_fraction') and model.early_stopping:
                ax.axvline(model.best_n_iter_, color='#f44336', linestyle='--', alpha=0.6, label=f'Early stop @ {model.best_n_iter_}')
            ax.set_xlabel("Época", color='#888')
            ax.set_ylabel("Log-Loss", color='#888')
            ax.tick_params(colors='#888')
            ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#ccc')
            for spine in ax.spines.values():
                spine.set_edgecolor('#2a2a2a')
            ax.grid(axis='y', color='#1f1f1f', linewidth=0.8)
            st.pyplot(fig_loss, use_container_width=True)
            plt.close()

    with tab3:
        st.markdown('<div class="section-header">Matriz de Confusión (conjunto de prueba)</div>', unsafe_allow_html=True)

        cm_data = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(10, 7))
        fig_cm.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')

        cmap = sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(
            cm_data, annot=True, fmt='d', cmap=cmap,
            xticklabels=CLASS_NAMES_SHORT,
            yticklabels=CLASS_NAMES_SHORT,
            ax=ax, cbar=True,
            linewidths=0.5, linecolor='#1a1a1a'
        )
        ax.tick_params(colors='#ccc', labelsize=8)
        ax.set_xlabel("Predicho", color='#888')
        ax.set_ylabel("Real", color='#888')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

        # Per-class report
        st.markdown('<div class="section-header">Reporte por clase</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES_SHORT, output_dict=True)
        import pandas as pd
        df_report = pd.DataFrame(report).T.iloc[:-3, :3].round(3)
        st.dataframe(df_report.style.background_gradient(cmap='YlOrBr', axis=None), use_container_width=True)

else:
    # Welcome state
    st.markdown("""
    <div class="info-box" style="padding:2rem;margin-top:1rem;">
        <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e8c547;margin-bottom:1rem;">
            ⬅ Configura y entrena tu red
        </div>
        <p>1. Elige el <strong>tipo de red</strong>: MLP o DNN</p>
        <p>2. Ajusta <strong>capas, neuronas y activación</strong></p>
        <p>3. Presiona <strong>Entrenar Modelo</strong></p>
        <p>4. Selecciona una imagen y clasifícala en tiempo real</p>
    </div>
    """, unsafe_allow_html=True)

    # Show sample grid
    st.markdown('<div class="section-header">Vista previa del dataset</div>', unsafe_allow_html=True)
    sample_indices = np.random.choice(len(X), 20, replace=False)
    fig_grid, axes = plt.subplots(2, 10, figsize=(14, 3))
    fig_grid.patch.set_facecolor('#0e0e0e')
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[sample_indices[i]].reshape(28, 28), cmap='inferno')
        ax.axis('off')
        ax.set_title(CLASS_NAMES_SHORT[y[sample_indices[i]]], 
                    fontsize=5.5, color='#888', pad=2, fontfamily='monospace')
    plt.tight_layout(pad=0.3)
    st.pyplot(fig_grid, use_container_width=True)
    plt.close()
