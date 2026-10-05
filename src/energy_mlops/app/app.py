import io
import json
import os
import time
import zipfile

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
def load_monitoring_artifacts():
    """Carrega o resumo concluído mais recente e seu HTML correspondente."""
    client = boto3.client(
        "s3",
        endpoint_url=settings.RUSTFS_ENDPOINT,
        aws_access_key_id=settings.RUSTFS_ROOT_USER,
        aws_secret_access_key=settings.RUSTFS_ROOT_PASSWORD,
        region_name="us-east-1",
    )
    bucket = settings.RUSTFS_BUCKET
    candidates = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix="monitoring/drift_report/",
    ):
        candidates.extend(
            obj for obj in page.get("Contents", [])
            if obj["Key"].endswith(".json")
        )
    if not candidates:
        return None
    latest = max(candidates, key=lambda obj: (obj["LastModified"], obj["Key"]))
    summary_bytes = client.get_object(
        Bucket=bucket, Key=latest["Key"],
    )["Body"].read()
    summary = json.loads(summary_bytes)
    # A chave histórica pareada evita combinar HTML e JSON de execuções distintas.
    html_key = latest["Key"].removesuffix(".json") + ".html"
    html_bytes = client.get_object(Bucket=bucket, Key=html_key)["Body"].read()
    comparison = summary.get("generation_comparison", {})
    hourly_bytes = daily_bytes = None
    hourly = daily = None
    if comparison:
        prefix = latest["Key"].removesuffix(".json")
        hourly_bytes = client.get_object(Bucket=bucket, Key=prefix + ".hourly.csv")["Body"].read()
        daily_bytes = client.get_object(Bucket=bucket, Key=prefix + ".daily.csv")["Body"].read()
        hourly = pd.read_csv(io.BytesIO(hourly_bytes), dtype={"model_version": str})
        daily = pd.read_csv(io.BytesIO(daily_bytes))
        if not hourly["execution_id"].eq(summary["execution_id"]).all():
            raise ValueError("Artefato horário pertence a outra execução.")
        if not hourly["model_run_id"].eq(summary["model"]["run_id"]).all():
            raise ValueError("Artefato horário pertence a outro modelo.")
        hourly["date_local"] = pd.to_datetime(hourly["date"], utc=True).dt.tz_convert("America/Sao_Paulo")
        hourly["dia_local"] = hourly["date_local"].dt.strftime("%Y-%m-%d")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("relatorio.html", html_bytes)
        package.writestr("resumo.json", summary_bytes)
        if hourly_bytes is not None:
            package.writestr("comparacao_horaria.csv", hourly_bytes)
            package.writestr("resumo_diario.csv", daily_bytes)
    return summary, html_bytes.decode("utf-8"), archive.getvalue(), hourly, daily


def monitoring_window_text(window):
    start = pd.Timestamp(window["start_utc"]).tz_convert("America/Sao_Paulo")
    end = pd.Timestamp(window["end_utc"]).tz_convert("America/Sao_Paulo")
    return (
        f"{start:%d/%m/%Y %H:%M} → {end:%d/%m/%Y %H:%M} "
        f"(UTC−03) | {window['rows']} horas"
    )

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

                executive_text = (
                    "💼 **Visão executiva — desempenho histórico**\n\n"
                    "Na avaliação histórica em um período posterior "
                    "ao treinamento (OOT), o erro médio normalizado "
                    f"foi de **{champion_nmae:.2f}% da capacidade instalada**. "
                )

                if champion_mae_mw is not None:
                    executive_text += (
                        "Em termos de potência, a diferença absoluta média "
                        "entre a geração prevista e a observada foi de "
                        f"**{champion_mae_mw:.0f} MW**. "
                    )

                executive_text += (
                    "Esses indicadores ajudam a dimensionar os desvios "
                    "das previsões usadas no planejamento da geração.\n\n"
                    "Esse resultado descreve o desempenho histórico. "
                    "A qualidade das previsões atuais será acompanhada "
                    "conforme os dados de geração observada estiverem "
                    "disponíveis."
                )

                st.success(executive_text)

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
    st.markdown("### 📉 Monitoramento de Dados e Desempenho")
    st.caption(
        "Mudanças nas variáveis de entrada e desempenho histórico do modelo "
        "na janela monitorada, comparado à referência OOT."
    )
    if st.button("🔄 Atualizar monitoramento"):
        load_monitoring_artifacts.clear()

    try:
        monitoring = load_monitoring_artifacts()
    except Exception as error:  # noqa: BLE001
        monitoring = None
        st.error(f"Não foi possível carregar os artefatos de monitoramento: {error}")

    if monitoring is None:
        st.info("Nenhuma execução completa disponível para exibição. Confira os logs do pipeline.")
    else:
        summary, html_content, archive_bytes, hourly, daily = monitoring
        reference = summary["reference_window"]
        current = summary["current_window"]
        performance = summary["performance"]
        model = summary["model"]
        evaluated = performance.get("evaluated_rows", 0)
        rows = current["rows"]
        evaluated_at = pd.Timestamp(summary["evaluated_at_utc"]).tz_convert("America/Sao_Paulo")

        st.markdown("#### Contexto da execução")
        st.write(f"**Referência:** {monitoring_window_text(reference)}")
        st.write(f"**Janela monitorada:** {monitoring_window_text(current)}")
        st.caption(
            f"Gerado em {evaluated_at:%d/%m/%Y %H:%M:%S} (UTC−03) | "
            f"Modelo avaliado: v{model['version']} | Execução: {summary['execution_id']}"
        )
        col_data, col_truth, col_performance = st.columns(3)
        col_data.metric("Variáveis com drift", f"{summary['data_drift']['share']:.1%}")
        col_truth.metric("Horas avaliadas com ground truth", f"{evaluated}/{rows}")
        current_nmae = performance.get("current_nmae_pct")
        delta = performance.get("delta_nmae_pp")
        col_performance.metric(
            "nMAE na janela avaliada",
            f"{current_nmae:.2f}%" if current_nmae is not None else "Indisponível",
            delta=f"{delta:+.2f} p.p. vs. OOT" if delta is not None else None,
            delta_color="inverse",
        )
        if current["window_status"] != "COMPLETE_MONTH" or evaluated < rows:
            st.info(
                "Janela parcial: o drift de dados considera a janela meteorológica; "
                "a performance considera somente horas com geração observada e "
                "capacidade válidas. A comparação com um mês completo é um diagnóstico inicial."
            )
        if current_nmae is not None:
            baseline = performance["baseline_nmae_pct"]
            threshold = summary["performance_threshold_pp"]
            result = "acionado" if performance["drift_detected"] else "não acionado"
            st.caption(
                f"Baseline OOT: {baseline:.2f}% | Alerta de performance: {result} | "
                f"Limiar de aumento: {threshold:.2f} p.p."
            )
        st.caption(
            "Mudanças nas variáveis de mês podem refletir a passagem do calendário. "
            "Data drift, isoladamente, não comprova perda de desempenho."
        )
        st.write(
            "**Retreinamento:** "
            + {
                "NOT_TRIGGERED": "não executado",
                "STARTED": "iniciado",
                "COMPLETED": "concluído",
                "FAILED": "falhou",
            }.get(summary["training_status"], summary["training_status"])
        )
        st.download_button(
            "📥 Baixar monitoramento completo (ZIP)",
            data=archive_bytes,
            file_name=f"monitoramento_{summary['execution_id']}.zip",
            mime="application/zip",
        )
        st.caption(
            "O ZIP contém o relatório interativo completo e o resumo da execução "
            "com datas, cobertura, modelo, métricas e caminhos dos dados. "
            "Quando disponíveis, inclui também a comparação horária e o resumo diário em CSV. "
            "Os datasets Parquet não estão incluídos."
        )
        with st.expander("Detalhes da execução e origem dos dados"):
            st.json(summary)
        if hourly is not None:
            st.markdown("#### Estimativa com meteorologia observada × geração observada")
            st.caption(
                "Avaliação retrospectiva do modelo desta execução, não das previsões "
                "day-ahead emitidas anteriormente. Horários em UTC−03."
            )
            days = hourly["dia_local"].drop_duplicates().tolist()
            selected_day = st.selectbox(
                "Período dos gráficos", ["Toda a janela", *days],
                key=f"monitoring_day_{summary['execution_id']}",
            )
            plotted = hourly if selected_day == "Toda a janela" else hourly.loc[hourly["dia_local"].eq(selected_day)]
            missing = int(plotted["observed_mw"].isna().sum())
            if missing:
                st.info(f"{missing} horas sem geração observada válida neste período. Lacunas não representam geração zero.")
            # Datas locais sem timezone apenas para rótulos Plotly; UTC preservado no CSV.
            x = plotted["date_local"].dt.tz_localize(None)
            generation = go.Figure()
            for column, label, color in [
                ("predicted_mw", "Estimativa do modelo", "#38BDF8"),
                ("observed_mw", "Geração observada (ONS)", "#10B981"),
            ]:
                generation.add_trace(go.Scatter(
                    x=x, y=plotted[column], name=label, mode="lines",
                    connectgaps=False, line={"color": color, "width": 2},
                    hovertemplate="%{x|%d/%m %H:%M}<br>%{y:,.1f} MW<extra>%{fullData.name}</extra>",
                ))
            generation.update_layout(
                template="plotly_dark", height=350, hovermode="x unified",
                xaxis_title="Horário local (UTC−03)", yaxis_title="Potência (MW)",
                legend={"orientation": "h", "y": 1.12},
                margin={"l": 20, "r": 20, "t": 45, "b": 30},
            )
            st.plotly_chart(generation, width="stretch")
            errors = go.Figure(go.Bar(
                x=x, y=plotted["error_mw"],
                marker_color=["#F59E0B" if pd.notna(value) and value >= 0 else "#A78BFA" for value in plotted["error_mw"]],
                hovertemplate="%{x|%d/%m %H:%M}<br>Erro: %{y:+,.1f} MW<extra></extra>",
            ))
            errors.add_hline(y=0, line_color="#94A3B8", line_width=1)
            errors.update_layout(
                template="plotly_dark", height=240,
                title="Erro horário: estimativa − observado",
                xaxis_title="Horário local (UTC−03)", yaxis_title="Erro (MW)",
                margin={"l": 20, "r": 20, "t": 45, "b": 30},
            )
            st.plotly_chart(errors, width="stretch")
            st.caption("Acima de zero: superestimação. Abaixo de zero: subestimação. Sem truth, não há erro calculado.")
            st.markdown("##### Resumo diário")
            displayed = daily.rename(columns={
                "dia_local": "Dia", "horas_meteorologicas": "Horas meteorológicas",
                "horas_truth_validas": "Horas com truth", "horas_esperadas": "Horas esperadas",
                "status": "Cobertura", "mae_mw": "MAE (MW)",
                "nmae_pct": "nMAE (%)", "bias_mw": "Viés médio (MW)",
            })
            st.dataframe(displayed, hide_index=True, width="stretch")
            st.caption("Métricas diárias são descritivas; o limiar de performance se aplica à janela agregada.")
        else:
            st.info("Esta execução histórica não contém comparação horária. Execute novamente o pipeline atualizado para gerar os gráficos.")
        with st.expander("Relatório detalhado de data drift (Evidently)", expanded=False):
            components.html(html_content, height=900, scrolling=True)
