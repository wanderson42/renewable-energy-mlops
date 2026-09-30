import os
import time

import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from energy_mlops.config import settings
from energy_mlops.service.xai_artifacts import load_xai_artifacts_from_mlflow

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
def load_cached_xai_artifacts(
    run_id: str,
):
    return load_xai_artifacts_from_mlflow(
        run_id
    )

@st.cache_data(ttl=60)
def fetch_model_info(api_base_url: str):
    response = requests.get(
        f"{api_base_url}/model-info",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()

@st.cache_data(ttl=1800)  # Cache de 30 minutos
def fetch_real_weather_forecast():
    """Busca a previsão real das próximas 24h da Open-Meteo para Morro do Chapéu - BA."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -11.5503,
        "longitude": -41.1565,
        "hourly": ["wind_speed_100m", "wind_direction_100m", "temperature_2m"],
        "timezone": "America/Maceio",
        "forecast_days": 2
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

    now = pd.Timestamp.now(tz="America/Maceio").tz_localize(None)
    df_forecast = df_forecast[df_forecast["date"] >= now].head(24)
    df_forecast["date"] = df_forecast["date"].dt.tz_localize("America/Maceio").dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {"predictions": df_forecast.to_dict(orient="records")}

@st.cache_data(ttl=300)
def load_drift_report_html():
    """Busca o relatório HTML gerado pelo Evidently AI diretamente no Data Lake (RustFS)."""
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=settings.RUSTFS_ENDPOINT,
            aws_access_key_id=settings.RUSTFS_ROOT_USER,
            aws_secret_access_key=settings.RUSTFS_ROOT_PASSWORD,
            region_name="us-east-1",
        )
        response = s3_client.get_object(Bucket="energy-lake", Key="monitoring/drift_report.html")
        return response['Body'].read().decode('utf-8')
    except Exception as e:  # noqa: BLE001, F841
        return None

# ==========================================
# BARRA LATERAL (SIDEBAR)
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/wind-turbine.png", width=80)
st.sidebar.title("Configurações do Serviço")

DEFAULT_API_BASE_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
)

api_base_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=DEFAULT_API_BASE_URL,
)

prediction_url = (
    f"{api_base_url}/predict/batch"
)

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

# Divisão Arquitetural em Abas Principais
main_tab_op, main_tab_drift = st.tabs(["🚀 Operação Day-Ahead & XAI", "📉 Monitoramento de Drift (Evidently)"])

# ------------------------------------------
# ABA 1: OPERAÇÃO E EXPLICABILIDADE
# ------------------------------------------
with main_tab_op:

    # ==========================================
    # 1. METADADOS OPERACIONAIS DO CHAMPION
    #    Fonte: FastAPI /model-info
    # ==========================================
    try:
        model_info = fetch_model_info(
            api_base_url
        )
    except Exception as e:  # noqa: BLE001
        model_info = None

        st.sidebar.warning(
            f"Não foi possível consultar /model-info: {e}"
        )

    # Valores default para evitar variáveis não definidas
    champion_version = None
    champion_run_id = None
    champion_nmae = None
    champion_mae_mw = None
    champion_r2 = None

    if model_info is not None:
        champion_version = model_info.get(
            "version"
        )

        champion_run_id = model_info.get(
            "run_id"
        )

        model_metrics = model_info.get(
            "metrics",
            {},
        )

        champion_nmae = model_metrics.get(
            "oot_nmae_pct"
        )

        champion_mae_mw = model_metrics.get(
            "oot_mae_mw"
        )

        champion_r2 = model_metrics.get(
            "oot_r2_score"
        )

    # ==========================================
    # 2. ARTEFATOS XAI
    #
    # O run_id vem da FastAPI.
    # O MLflow é usado apenas como Artifact Store.
    # ==========================================
    if champion_run_id is not None:
        (
            explainability_dir,
            model_summary,
            xai_error,
        ) = load_cached_xai_artifacts(
            champion_run_id
        )

    else:
        explainability_dir = None
        model_summary = None
        xai_error = (
            "Run ID do Champion indisponível."
        )

    # ==========================================
    # 3. PREVISÃO METEOROLÓGICA
    # ==========================================
    try:
        payload = fetch_real_weather_forecast()

    except Exception as e:  # noqa: BLE001
        st.error(
            f"Erro ao buscar previsão da Open-Meteo: {e}"
        )
        st.stop()

    # ==========================================
    # 4. BOTÃO DE INFERÊNCIA
    # ==========================================
    col_btn, _ = st.columns(
        [2, 5]
    )

    with col_btn:
        run_prediction = st.button(
            "🚀 Executar Previsão Day-Ahead (24h)",
            type="primary",
            width="stretch",
        )

    # ==========================================
    # 5. EXECUÇÃO / RECUPERAÇÃO DA ÚLTIMA
    #    PREVISÃO
    # ==========================================
    if (
        run_prediction
        or "last_response" in st.session_state
    ):

        if run_prediction:

            with st.spinner(
                "Conectando à API de Inferência "
                "no Kubernetes..."
            ):
                try:
                    start_time = time.time()

                    response = requests.post(
                        prediction_url,
                        json=payload,
                        timeout=10,
                    )

                    latency_ms = (
                        time.time() - start_time
                    ) * 1000

                    if response.status_code == 200:

                        st.session_state[
                            "last_response"
                        ] = response.json()

                        st.session_state[
                            "latency"
                        ] = latency_ms

                        st.success(
                            "Inferência em lote "
                            "concluída com sucesso!"
                        )

                    else:
                        st.error(
                            f"Erro na API "
                            f"({response.status_code}): "
                            f"{response.text}"
                        )

                        st.stop()

                except Exception as e:  # noqa: BLE001
                    st.error(
                        "Falha ao conectar com o serviço "
                        f"em {prediction_url}: {e}"
                    )

                    st.stop()

        # ==========================================
        # 6. RESULTADOS DA INFERÊNCIA
        # ==========================================
        results = st.session_state[
            "last_response"
        ]

        latency_ms = st.session_state.get(
            "latency",
            0.0,
        )

        df_res = pd.DataFrame(
            results
        )

        df_res["date"] = pd.to_datetime(
            df_res["date"]
        )

        df_inputs = pd.DataFrame(
            payload["predictions"]
        )

        df_inputs["date"] = pd.to_datetime(
            df_inputs["date"]
        )

        df_merged = pd.merge(
            df_res,
            df_inputs,
            on="date",
        )

        # ==========================================
        # 7. SIDEBAR — GOVERNANÇA MLOps
        # ==========================================
        st.sidebar.markdown(
            "---"
        )

        st.sidebar.subheader(
            "⚙️ Monitoramento MLOps"
        )

        st.sidebar.metric(
            "Latência da API",
            f"{latency_ms:.0f} ms",
            delta="FastAPI",
            delta_color="normal",
        )

        # ------------------------------------------
        # Modelo Champion
        # ------------------------------------------
        if champion_version is not None:

            st.sidebar.metric(
                "Modelo Ativo",
                f"v{champion_version}",
                delta="@champion",
                delta_color="normal",
            )

        else:
            st.sidebar.metric(
                "Modelo Ativo",
                "Indisponível",
            )

        # ------------------------------------------
        # Terceiro indicador: nMAE OOT
        # ------------------------------------------
        if champion_nmae is not None:

            st.sidebar.metric(
                "nMAE OOT",
                f"{champion_nmae:.2f}%",
                help=(
                    "Erro absoluto médio normalizado "
                    "do modelo Champion no conjunto OOT. "
                    "Quanto menor, melhor."
                ),
            )

        # ==========================================
        # 8. KPIs OPERACIONAIS DAY-AHEAD
        # ==========================================
        st.markdown(
            "### 📊 Indicadores Globais de Geração "
            "(Próximas 24h)"
        )

        kpi1, kpi2, kpi3, kpi4 = (
            st.columns(4)
        )

        total_energy_gwh = (
            df_merged["predicted_mw"].sum()
            / 1000
        )

        avg_fc = (
            df_merged["predicted_fc"].mean()
            * 100
        )

        max_mw = df_merged[
            "predicted_mw"
        ].max()

        avg_wind = df_merged[
            "wind_speed_100m"
        ].mean()

        kpi1.metric(
            "Geração Total Est.",
            f"{total_energy_gwh:.2f} GWh",
            delta="24h Acumulado",
        )

        kpi2.metric(
            "Fator de Capacidade Médio",
            f"{avg_fc:.1f}%",
        )

        kpi3.metric(
            "Pico de Potência",
            f"{max_mw:.0f} MW",
        )

        kpi4.metric(
            "Velocidade Média do Vento",
            f"{avg_wind:.1f} m/s",
        )

        st.markdown(
            "---"
        )

        # ==========================================
        # 9. GRÁFICOS OPERACIONAIS
        # ==========================================
        col_chart1, col_chart2 = (
            st.columns(2)
        )

        # ------------------------------------------
        # Curva de geração
        # ------------------------------------------
        with col_chart1:

            st.subheader(
                "Curva de Geração Preditiva (MW)"
            )

            fig_mw = px.line(
                df_merged,
                x="date",
                y="predicted_mw",
                markers=True,
                labels={
                    "date": "Horário",
                    "predicted_mw": "Potência (MW)",
                },
                template="plotly_dark",
            )

            fig_mw.update_traces(
                line_color="#10B981",
                line_width=3,
            )

            st.plotly_chart(
                fig_mw,
                width="stretch",
            )

        # ------------------------------------------
        # Vento x fator de capacidade
        # ------------------------------------------
        with col_chart2:

            st.subheader(
                "Vento (100m) vs. "
                "Fator de Capacidade (%)"
            )

            fig_dual = go.Figure()

            fig_dual.add_trace(
                go.Bar(
                    x=df_merged["date"],
                    y=df_merged[
                        "wind_speed_100m"
                    ],
                    name="Vento (m/s)",
                    marker_color="#3B82F6",
                    opacity=0.6,
                )
            )

            fig_dual.add_trace(
                go.Scatter(
                    x=df_merged["date"],
                    y=(
                        df_merged[
                            "predicted_fc"
                        ]
                        * 100
                    ),
                    name="FC (%)",
                    yaxis="y2",
                    line={
                        "color": "#F59E0B",
                        "width": 3,
                    },
                )
            )

            fig_dual.update_layout(
                template="plotly_dark",
                yaxis={
                    "title": "Vento (m/s)"
                },
                yaxis2={
                    "title": "FC (%)",
                    "overlaying": "y",
                    "side": "right",
                },
                legend={
                    "x": 0,
                    "y": 1.1,
                    "orientation": "h",
                },
            )

            st.plotly_chart(
                fig_dual,
                width="stretch",
            )

        st.markdown(
            "---"
        )

        # ==========================================
        # 10. TRANSPARÊNCIA E XAI
        # ==========================================
        st.markdown(
            "### 🔍 Explicabilidade & "
            "Transparência do Modelo (SHAP)"
        )

        if (
            explainability_dir
            and os.path.exists(
                explainability_dir
            )
        ):

            # ======================================
            # Arquitetura
            #
            # Fonte atual:
            # model_summary.json no MLflow/RustFS
            # ======================================
            if model_summary:

                arch = model_summary.get(
                    "architecture",
                    "Ensemble",
                )

                meta = model_summary.get(
                    "meta_learner",
                    "LinearRegression",
                )

                weights = model_summary.get(
                    "meta_weights",
                    {},
                )

                weights_str = ", ".join(
                    [
                        f"{name}: {weight}"
                        for name, weight
                        in weights.items()
                    ]
                )

                weights_fmt = (
                    f" [{weights_str}]"
                    if weights_str
                    else ""
                )

                # Versão e R² agora vêm da API
                version_text = (
                    f"v{champion_version}"
                    if champion_version is not None
                    else "N/A"
                )

                r2_text = (
                    f"{champion_r2:.4f}"
                    if champion_r2 is not None
                    else "N/A"
                )

                caption_text = (
                    f"Exibindo diagnósticos SHAP do modelo "
                    f"**`{arch}`** "
                    f"({meta}{weights_fmt}) | "
                    f"**R² OOT:** {r2_text} | "
                    f"**Versão:** {version_text} "
                    f"(`@champion`)"
                )

            else:

                version_text = (
                    f"v{champion_version}"
                    if champion_version is not None
                    else "N/A"
                )

                caption_text = (
                    "Exibindo diagnósticos SHAP "
                    "do modelo **`@champion`** "
                    f"(Versão {version_text})"
                )

            st.caption(
                f"{caption_text} via MLflow/RustFS."
            )

            # ======================================
            # 11. DESEMPENHO OOT
            #
            # Métricas vêm da FastAPI /model-info
            # ======================================
            if champion_nmae is not None:

                st.info(
                    "📐 **Desempenho de "
                    "Generalização — OOT**\n\n"
                    f"O modelo **Champion** apresentou "
                    f"**nMAE OOT de "
                    f"{champion_nmae:.2f}%**, "
                    "indicando que o erro absoluto médio "
                    "correspondeu a aproximadamente "
                    f"**{champion_nmae:.1f}% da capacidade "
                    "utilizada na normalização**."
                )

            # Métricas técnicas complementares
            if (
                champion_nmae is not None
                or champion_mae_mw is not None
                or champion_r2 is not None
            ):

                metric_parts = []

                if champion_nmae is not None:
                    metric_parts.append(
                        f"**nMAE:** "
                        f"{champion_nmae:.2f}%"
                    )

                if champion_mae_mw is not None:
                    metric_parts.append(
                        f"**MAE:** "
                        f"{champion_mae_mw:.2f} MW"
                    )

                if champion_r2 is not None:
                    metric_parts.append(
                        f"**R²:** "
                        f"{champion_r2:.4f}"
                    )

                st.caption(
                    " | ".join(
                        metric_parts
                    )
                )

            st.info(
                "💡 **Observação:** Utilize a barra "
                "de ferramentas no canto superior direito "
                "de cada imagem para dar zoom (+ e -), "
                "arrastar (Pan) e restaurar o tamanho "
                "original."
            )

            # ======================================
            # 12. ABAS SHAP
            # ======================================
            (
                tab_summary,
                tab_dependence,
                tab_waterfall,
                tab_csv,
            ) = st.tabs(
                [
                    "📊 Summary Plot",
                    "📈 Dependence Plot",
                    "💧 Waterfall Plot",
                    "🔢 Ranking de Importância (CSV)",
                ]
            )

            summary_path = os.path.join(
                explainability_dir,
                "shap_summary.png",
            )

            dependence_path = os.path.join(
                explainability_dir,
                "shap_dependence.png",
            )

            waterfall_path = os.path.join(
                explainability_dir,
                "shap_waterfall.png",
            )

            csv_path = os.path.join(
                explainability_dir,
                "shap_importance.csv",
            )

            def plot_interactive_image(
                img_path
            ):
                img = Image.open(
                    img_path
                )

                fig = px.imshow(
                    img
                )

                fig.update_layout(
                    height=550,
                    coloraxis_showscale=False,
                    margin={
                        "l": 25,
                        "r": 25,
                        "t": 25,
                        "b": 25,
                    },
                    xaxis_visible=False,
                    yaxis_visible=False,
                    hovermode=False,
                    dragmode="zoom",
                )

                return fig

            # --------------------------------------
            # Summary Plot
            # --------------------------------------
            with tab_summary:

                if os.path.exists(
                    summary_path
                ):
                    st.plotly_chart(
                        plot_interactive_image(
                            summary_path
                        ),
                        width="stretch",
                    )

                    st.caption(
                        "SHAP Summary Plot: "
                        "Contribuição de cada feature "
                        "para o Fator de Capacidade."
                    )

            # --------------------------------------
            # Dependence Plot
            # --------------------------------------
            with tab_dependence:

                if os.path.exists(
                    dependence_path
                ):
                    st.plotly_chart(
                        plot_interactive_image(
                            dependence_path
                        ),
                        width="stretch",
                    )

                    st.caption(
                        "SHAP Dependence Plot: "
                        "Não-linearidades aprendidas "
                        "pelas árvores."
                    )

            # --------------------------------------
            # Waterfall Plot
            # --------------------------------------
            with tab_waterfall:

                if os.path.exists(
                    waterfall_path
                ):
                    st.plotly_chart(
                        plot_interactive_image(
                            waterfall_path
                        ),
                        width="stretch",
                    )

                    st.caption(
                        "SHAP Waterfall Plot: "
                        "Demonstração da decomposição "
                        "matemática das features para "
                        "uma hora específica."
                    )

            # --------------------------------------
            # Ranking CSV
            # --------------------------------------
            with tab_csv:

                if os.path.exists(
                    csv_path
                ):
                    df_shap_imp = pd.read_csv(
                        csv_path
                    )

                    st.dataframe(
                        df_shap_imp,
                        width="stretch",
                    )

        else:
            st.info(
                "Não foi possível carregar os "
                "artefatos de explicabilidade "
                f"da Run Champion: {xai_error}"
            )

        # ==========================================
        # 13. RESPOSTA BRUTA DA API
        # ==========================================
        with st.expander(
            "📄 Visualizar Tabela de "
            "Resposta do Servidor JSON"
        ):

            st.dataframe(
                df_merged[
                    [
                        "date",
                        "predicted_fc",
                        "predicted_mw",
                        "wind_speed_100m",
                        "wind_direction_100m",
                        "temperature_2m",
                    ]
                ],
                width="stretch",
            )

# ------------------------------------------
# ABA 2: MONITORAMENTO (EVIDENTLY AI)
# ------------------------------------------
with main_tab_drift:
    st.markdown(
        "### 📉 Monitoramento de Degradação "
        "(Data Drift & Performance Drift)"
    )

    st.caption(
        "O pipeline monitora mudanças estatísticas "
        "nas features do modelo e degradação do nMAE "
        "do Champion em relação à baseline OOT."
    )

    st.caption(
        "Painel interativo gerado pelo **Evidently AI**. " \
        "Exibe divergências estatísticas entre os dados de " \
        "validação (Reference) e os dados de produção recentes (Current)."
        )

    html_content = load_drift_report_html()

    if html_content:
        # Renderiza o HTML do Evidently preenchendo o espaço da aba
        components.html(html_content, height=900, scrolling=True)

        # Botão de ação (Mockup) para um Engenheiro MLOps decidir agir sobre o Drift
        if st.button("🔄 Disparar Pipeline de Treinamento Contínuo (CT)", help="Invoca manualmente o flow de retreinamento no Prefect"):
            st.warning("No ambiente de Portfólio atual, essa ação é demonstrativa. Em produção, isso acionaria a API REST do Prefect.")
    else:
        st.warning("⚠️ Relatório de Drift não encontrado no Data Lake (RustFS).")
        st.info("💡 **Ação Necessária:** Execute o script `poetry run python src/energy_mlops/pipelines/monitoring_flow.py` para calcular o Drift e gerar o artefato HTML.")