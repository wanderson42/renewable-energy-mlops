import json
import os
import time

#from datetime import datetime, timedelta
import boto3

#import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

#from pytz import UTC
from energy_mlops.config import settings

# 2. Injeta variáveis de ambiente no processo do OS
os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = settings.RUSTFS_ROOT_USER
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.RUSTFS_ROOT_PASSWORD
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
os.environ["AWS_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
os.environ["AWS_ENDPOINT_URL_S3"] = settings.RUSTFS_ENDPOINT

# 3. Boto3 usa as credenciais do RustFS para baixar artefatos do MLflow
boto3.setup_default_session(
    aws_access_key_id=settings.RUSTFS_ROOT_USER,
    aws_secret_access_key=settings.RUSTFS_ROOT_PASSWORD,
    region_name="us-east-1",
)

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"

# Configuração da Página Streamlit
st.set_page_config(
    page_title="Wind Forecast Dashboard | Bahia",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS leve
st.markdown(
    """
    <style>
    .main-metric {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# FUNÇÕES DE CACHE E EXTRAÇÃO DE DADOS
# ==========================================

@st.cache_data(ttl=60)
def load_shap_artifacts_from_mlflow():
    """Busca dinamicamente os artefatos de SHAP e o resumo do modelo @champion no MLflow."""
    try:
        boto3.setup_default_session(
            aws_access_key_id=settings.RUSTFS_ROOT_USER,
            aws_secret_access_key=settings.RUSTFS_ROOT_PASSWORD,
            region_name="us-east-1",
        )
        client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        model_version = client.get_model_version_by_alias("ensemble_lgb_xgb_rf_bahia", "champion")
        run = client.get_run(model_version.run_id)
        
        # URIs dos artefatos alvo
        shap_artifact_uri = f"{run.info.artifact_uri}/explainability"
        summary_artifact_uri = f"{run.info.artifact_uri}/model_summary.json"
        
        # Download da pasta de imagens
        local_explainability_dir = mlflow.artifacts.download_artifacts(artifact_uri=shap_artifact_uri)
        
        # Tenta baixar e ler o JSON gerado dinamicamente no treino
        try:
            local_summary_path = mlflow.artifacts.download_artifacts(artifact_uri=summary_artifact_uri)
            with open(local_summary_path, "r") as f:
                model_summary = json.load(f)
        except Exception:  # noqa: BLE001
            model_summary = None

        return local_explainability_dir, model_version.version, model_summary
    except Exception as e:  # noqa: BLE001
        return None, str(e), None


@st.cache_data(ttl=1800)  # Cache de 30 minutos
def fetch_real_weather_forecast():
    """Busca a previsão real das próximas 24h da Open-Meteo para Morro do Chapéu - BA."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -11.5503,
        "longitude": -41.1565,
        "hourly": ["wind_speed_100m", "wind_direction_100m", "temperature_2m"],
        "timezone": "America/Maceio",
        "forecast_days": 2 # Garante tempo suficiente para fatiar 24h a partir de "agora"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    hourly = data["hourly"]
    df_forecast = pd.DataFrame({
        "date": pd.to_datetime(hourly["time"]),
        "wind_speed_100m": hourly["wind_speed_100m"],
        "wind_direction_100m": hourly["wind_direction_100m"],
        "temperature_2m": hourly["temperature_2m"]
    })
    
    # Filtra apenas as próximas 24 horas a partir do momento atual
    now = pd.Timestamp.now(tz="America/Maceio").tz_localize(None)
    df_forecast = df_forecast[df_forecast["date"] >= now].head(24)
    
    # Converte a data para o padrão ISO UTC exigido pelo contrato da API (schema.py)
    df_forecast["date"] = df_forecast["date"].dt.tz_localize("America/Maceio").dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return {"predictions": df_forecast.to_dict(orient="records")}


# ==========================================
# BARRA LATERAL (SIDEBAR)
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/wind-turbine.png", width=80)
st.sidebar.title("Configurações do Serviço")

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/predict/batch")
api_url = st.sidebar.text_input("Endpoint da API FastAPI", value=DEFAULT_API_URL)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Localização do Ativo (Open-Meteo)")
st.sidebar.text("Polo: Morro do Chapéu - BA")
st.sidebar.text("Lat: -11.5503 | Lon: -41.1565")
st.sidebar.caption("As previsões meteorológicas reais (24h) são carregadas automaticamente no startup.")


# ==========================================
# CORPO PRINCIPAL
# ==========================================
st.title("⚡ Painel de Previsão de Geração Eólica — Parque Bahia")
st.caption("Visualização operacional para despacho **Day-Ahead** alimentada por MLOps & FastAPI em Kubernetes.")

explainability_dir, version_or_err, model_summary = load_shap_artifacts_from_mlflow()


# Gera o payload real
try:
    payload = fetch_real_weather_forecast()
except Exception as e:  # noqa: BLE001
    st.error(f"Erro ao buscar previsão da Open-Meteo: {e}")
    st.stop()

col_btn, _ = st.columns([2, 5])
with col_btn:
    run_prediction = st.button("🚀 Executar Previsão Day-Ahead (24h)", type="primary", width="stretch")

if run_prediction or "last_response" in st.session_state:
    if run_prediction:
        with st.spinner("Conectando à API de Inferência no Kubernetes..."):
            try:
                # Dispara o cronômetro para medir a latência
                start_time = time.time()
                response = requests.post(api_url, json=payload, timeout=10)
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    st.session_state["last_response"] = response.json()
                    st.session_state["latency"] = latency_ms
                    st.success("Inferência em lote concluída com sucesso!")
                else:
                    st.error(f"Erro na API ({response.status_code}): {response.text}")
                    st.stop()
            except Exception as e:  # noqa: BLE001
                st.error(f"Falha ao conectar com o serviço em {api_url}: {e}")
                st.stop()

    # Leitura dos Resultados armazenados
    results = st.session_state["last_response"]
    latency_ms = st.session_state.get("latency", 0.0)
    
    df_res = pd.DataFrame(results)
    df_res["date"] = pd.to_datetime(df_res["date"])

    df_inputs = pd.DataFrame(payload["predictions"])
    df_inputs["date"] = pd.to_datetime(df_inputs["date"])
    
    # Junta as predições com as features originais para plotagem
    df_merged = pd.merge(df_res, df_inputs, on="date")
    
    # ==========================================
    # MÉTRICAS DE INFRAESTRUTURA MLOPS (SIDEBAR)
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Monitoramento MLOps")
    st.sidebar.metric("Latência da API", f"{latency_ms:.0f} ms", delta="FastAPI", delta_color="normal")
    st.sidebar.metric("Modelo Ativo", f"v{version_or_err}", delta="@champion", delta_color="normal")

    # ==========================================
    # MÉTRICAS DE ALTO NÍVEL (KPIs)
    # ==========================================
    st.markdown("### 📊 Indicadores Globais de Geração (Próximas 24h)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_energy_gwh = (df_merged["predicted_mw"].sum()) / 1000
    avg_fc = df_merged["predicted_fc"].mean() * 100
    max_mw = df_merged["predicted_mw"].max()
    avg_wind = df_merged["wind_speed_100m"].mean()

    kpi1.metric("Geração Total Est.", f"{total_energy_gwh:.2f} GWh", delta="24h Acumulado")
    kpi2.metric("Fator de Capacidade Médio", f"{avg_fc:.1f}%")
    kpi3.metric("Pico de Potência", f"{max_mw:.0f} MW")
    kpi4.metric("Velocidade Média do Vento", f"{avg_wind:.1f} m/s")

    st.markdown("---")

    # ==========================================
    # GRÁFICOS INTERATIVOS
    # ==========================================
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader(" Curva de Geração Preditiva (MW)")
        fig_mw = px.line(
            df_merged,
            x="date",
            y="predicted_mw",
            markers=True,
            labels={"date": "Horário", "predicted_mw": "Potência (MW)"},
            template="plotly_dark",
        )
        fig_mw.update_traces(line_color="#10B981", line_width=3)
        st.plotly_chart(fig_mw, width="stretch")

    with col_chart2:
        st.subheader(" Vento (100m) vs. Fator de Capacidade (%)")
        fig_dual = go.Figure()
        fig_dual.add_trace(
            go.Bar(
                x=df_merged["date"],
                y=df_merged["wind_speed_100m"],
                name="Vento (m/s)",
                marker_color="#3B82F6",
                opacity=0.6,
            )
        )
        fig_dual.add_trace(
            go.Scatter(
                x=df_merged["date"],
                y=df_merged["predicted_fc"] * 100,
                name="FC (%)",
                yaxis="y2",
                line={"color": "#F59E0B", "width": 3},
            )
        )
        fig_dual.update_layout(
            template="plotly_dark",
            yaxis={"title": "Vento (m/s)"},
            yaxis2={"title": "FC (%)", "overlaying": "y", "side": "right"},
            legend={"x": 0, "y": 1.1, "orientation": "h"},
        )
        st.plotly_chart(fig_dual, width="stretch")

    st.markdown("---")

    # ==========================================
    # EXPLICABILIDADE SHAP (XAI)
    # ==========================================
    st.markdown("### 🔍 Explicabilidade & Transparência do Modelo (SHAP)")

    if explainability_dir and os.path.exists(explainability_dir):
        if model_summary:
            arch = model_summary.get("architecture", "Ensemble")
            meta = model_summary.get("meta_learner", "LinearRegression")
            weights = model_summary.get("meta_weights", {})
            r2 = model_summary.get("metrics", {}).get("r2_score", "N/A")
            
            weights_str = ", ".join([f"{k}: {v}" for k, v in weights.items()])
            weights_fmt = f" [{weights_str}]" if weights_str else ""
            
            caption_text = (
                f"Exibindo diagnósticos SHAP do modelo **`{arch}`** ({meta}{weights_fmt}) | "
                f"**$R^2$ Validação:** {r2} | **Versão:** v{version_or_err} (`@champion`)"
            )
        else:
            caption_text = f"Exibindo diagnósticos SHAP extraídos do modelo **`@champion`** (Versão {version_or_err})"

        st.caption(f"{caption_text} via MLflow/RustFS.")
        # Explicabilidade & Transparência do Modelo (SHAP)
        st.info("💡 **Observação:** Utilize a barra de ferramentas no canto superior direito de cada imagem para dar zoom (+ e -), arrastar (Pan) e restaurar o tamanho original.")

        # ==========================================
        # TRADUÇÃO PARA VISÃO DE NEGÓCIOS
        # ==========================================
        if model_summary:
            metrics = model_summary.get("metrics", {})
            nmae_pct = metrics.get("oot_nmae_pct", 0.0)
            
            if nmae_pct > 0:
                forecast_accuracy = 100 - nmae_pct
                st.success(
                    f"💼 **Visão Executiva (Taxa de Acerto Global): {forecast_accuracy:.1f}%**\n\n"
                    f"O modelo atual apresenta um erro médio normalizado (nMAE) de apenas {nmae_pct:.2f}%. "
                    f"Isso significa que, na média de validação OOT, as nossas estimativas de geração de energia eólica "
                    f"refletem a realidade com **{forecast_accuracy:.1f}% de precisão global**."
                )

        tab_summary, tab_dependence, tab_waterfall, tab_csv = st.tabs([
            "📊 Summary Plot",
            "📈 Dependence Plot",
            "💧 Waterfall Plot",
            "🔢 Ranking de Importância (CSV)",
        ])

        summary_path = os.path.join(explainability_dir, "shap_summary.png")
        dependence_path = os.path.join(explainability_dir, "shap_dependence.png")
        waterfall_path = os.path.join(explainability_dir, "shap_waterfall.png")
        csv_path = os.path.join(explainability_dir, "shap_importance.csv")

        # Função auxiliar para renderizar imagem de alta qualidade com controles Plotly
        def plot_interactive_image(img_path):
            img = Image.open(img_path)
            fig = px.imshow(img)
            
            fig.update_layout(
                height=550,  # Aumenta a área de exibição (~15% maior que o padrão)
                coloraxis_showscale=False,
                margin={"l": 25, "r": 25, "t": 25, "b": 25},  # Margens expandidas para criar a folga
                xaxis_visible=False,
                yaxis_visible=False,
                hovermode=False,  # Desativa o tooltip de pixels
                dragmode="zoom"   # Mantém a ferramenta de zoom por seleção como padrão
            )
            return fig

        with tab_summary:
            if os.path.exists(summary_path):
                st.plotly_chart(plot_interactive_image(summary_path), width='stretch')
                st.caption("SHAP Summary Plot: Contribuição de cada feature para o Fator de Capacidade.")

        with tab_waterfall:
            if os.path.exists(waterfall_path):
                st.plotly_chart(plot_interactive_image(waterfall_path), width='stretch')
                st.caption("SHAP Waterfall Plot: Demonstração da decomposição matemática das features para uma hora específica. Ilustra o peso exato de variáveis como a Temperatura e o Vento sobre a predição final base (Expected Value).")

        with tab_dependence:
            if os.path.exists(dependence_path):
                st.plotly_chart(plot_interactive_image(dependence_path), width='stretch')
                st.caption("SHAP Dependence Plot: Não-linearidades aprendidas pelas árvores.")

        with tab_csv:
            if os.path.exists(csv_path):
                df_shap_imp = pd.read_csv(csv_path)
                st.dataframe(df_shap_imp, width='stretch')

    else:
        st.info(f"Não foi possível carregar os artefatos de explicabilidade do MLflow: {version_or_err}")

    # Tabela de Dados Brutos
    with st.expander("📄 Visualizar Tabela de Resposta do Servidor JSON"):
        st.dataframe(
            df_merged[["date", "predicted_fc", "predicted_mw", "wind_speed_100m", "wind_direction_100m", "temperature_2m"]],
            width="stretch",
        )