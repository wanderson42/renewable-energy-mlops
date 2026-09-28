## O repositório foi estruturado seguindo os padrões modernos de engenharia de software e MLOps. A gestão de dependências é feita via **Poetry** e o pacote principal é instalável localmente (`energy_mlops`).

```text
renewable-energy-mlops/
├── Dockerfile                         # Configuração da imagem do serviço FastAPI
├── Makefile                           # Automação (ports, stop-ports, test-opt, test-train)
├── pyproject.toml                     # Dependências (Poetry) e ferramentas de linting
├── poetry.lock                        # Lockfile de dependências exatas
├── tox.ini                            # Orquestrador de testes isolados para CI/CD
├── README.md                          # Documentação geral do repositório
├── lista_ordenada_de_comandos_bash.txt # Histórico de comandos e setup do ambiente
├── helm/                              # Infraestrutura Kubernetes (KinD)
│   ├── Chart.yaml           
│   ├── values.yaml          
│   └── templates/             
│       ├── api.yaml                   # Deploy e Service do Model Serving (FastAPI)
│       ├── rustfs.yaml                # Object Storage local (PVC + S3 MinIO compatible)
│       ├── mlflow.yaml                # Deployment do MLflow Tracking Server
│       └── postgres.yaml              # Banco de dados de metadados do MLflow (PVC)
├── notebooks/               
│   └── renewable-energy-mlops.ipynb   # Documentação e relatório técnico central
├── src/
│   └── energy_mlops/          
│       ├── __init__.py
│       ├── config.py                  # Pydantic Settings (Conexões S3, MLflow, etc.)
│       ├── app/
│       │   └── app.py                 # Interface / Dashboard de monitoramento
│       ├── data/              
│       │   ├── build_features.py      # Engenharia de atributos e merge temporal (pd.merge_asof)
│       │   ├── extract_energy.py      # Extração de dados de geração eólica
│       │   ├── extract_weather.py     # Extração de dados meteorológicos
│       │   ├── schema.py              # Validação de schema de entrada/saída (Pandera)
│       │   └── schema_data.py         # Schemas de dados processados
│       ├── models/            
│       │   ├── interfaces.py          # Contratos de interface (ModelTrainer, ModelOptimizer)
│       │   ├── optimize_stacking.py   # Otimização de hiperparâmetros com Optuna
│       │   ├── train_stacking.py      # Algoritmo principal (Stacking Ensemble LGBM+XGB+RF)
│       │   ├── train_examples.py      # Exemplos de referência e Mocks para testes (Dummy, Ridge, PyTorch)
│       │   └── train.py               # Scripts auxiliares de treino
│       ├── pipelines/         
│       │   ├── backfill_flow.py       # Pipeline Prefect de carga histórica de dados
│       │   ├── data_ingestion_flow.py # Pipeline Prefect de ingestão diária de dados
│       │   ├── training_flow.py       # Pipeline Prefect de Treinamento Contínuo (CT) agnóstico
│       │   └── utils.py               # Utilitários de Io e persistência no Data Lake
│       └── service/                   # Camada de Serviço (Model Serving)
│           ├── main.py                # Endpoint assíncrono FastAPI (/predict/batch e /reload-model)
│           └── schema.py              # Contratos Pydantic para validação das requisições HTTP
└── tests/                   
    ├── conftest.py                    # Fixtures globais do pytest
    ├── test_config.py                 # Testes de variáveis de ambiente e Pydantic Settings
    ├── data/
    │   ├── test_schema.py             # Testes de validação de schemas Pandera
    │   └── test_build_features.py     # Testes de engenharia de features (merge temporal)
    ├── models/
    │   └── test_heuristics.py         # Testes comportamentais e heurísticos do modelo
    └── service/
        ├── test_schema.py             # Testes de validação de payload Pydantic
        └── test_main.py               # Testes de integração das rotas HTTP da FastAPI
```



## Diagrama arquitetural com o host, o gerenciamento de rede e o cluster Kubernetes com os Volumes Persistentes (PVCs):

```text

 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       ECOSSISTEMA EXTERNO & CLOUD (GitHub)                                    │
 │                                                                                                               │
 │   [ Desenvolvedor / Push ] ────▶ [ GitHub Actions (CI: tox tests) ] ────▶ [ GHCR (Container Registry) ]       │
 └───────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                     │ Image Pull (Helm / GitOps)
                                                     ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  MÁQUINA HOST & GERENCIAMENTO DE REDE (Alienware-16)                          │
│                                                                                                               │
│   ┌─────────────────────────┐       ┌───────────────────────┐       ┌─────────────────────────────────────┐   │
│   │ Streamlit Dashboard     │       │ GERENCIAMENTO DE REDE │       │ AMBIENTE LOCAL / ORQUESTRAÇÃO       │   │
│   │ (app.py via Poetry)     │       │ ▹ make ports          │       │ ▹ Prefect CT Flows                  │   │
│   │ ▹ UI (:8501)            │       │   (port-forward 8000) │       │ ▹ Hot-Reload HTTP Trigger           │   │
│   └────────────┬────────────┘       └───────────┬───────────┘       └─────────────────────────────────────┘   │
└────────────────┼────────────────────────────────┼─────────────────────────────────────────────────────────────┘
                 │                                │ Túnel K8s
                 ▼                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                KUBERNETES CLUSTER (KinD - Docker Container)                                   │
│                                NODE: energy-mlops-control-plane                                               │
│                                                                                                               │
│   ┌─────────────────────────────┐     ┌──────────────────────────┐     ┌───────────────────────────────────┐  │
│   │ POD: api-deployment         │     │ POD: mlflow-tracking     │     │ POD: postgres-db                  │  │
│   │ (FastAPI + Hot-Reload)      │     │ (Tracking Server)        │     │ (Database do MLflow)              │  │
│   │ ▹ CONTAINER: energy-api     │     │ ▹ CONTAINER: mlflow      │     │ ▹ CONTAINER: postgres             │  │
│   │   (:8000 REST API)          │     │   (:5000 UI/API)         │     │   (:5432 TCP)                     │  │
│   └────────────┬────────────────┘     └────────────┬─────────────┘     └─────────────────┬─────────────────┘  │
│                │                                   │                                     │ (PVC)              │
│                │ Lê modelo @champion               │ Salva Metadados                     ▼                    │
│                ▼                                   │                           [ postgres-pvc:2Gi ]           │
│   ┌─────────────────────────────┐                  │                                                          │
│   │ POD: rustfs-storage         │◀─────────────────┘                                                          │
│   │ (Data Lake / S3 Compatible) │                                                                             │
│   │ ▹ CONTAINER: rustfs         │─────── (PVC) ──────▶ [ rustfs-pvc:5Gi ]                                     │
│   │   (:9000 S3 API)            │                      (Garante retenção do Data Lake e Modelos)              │
│   └─────────────────────────────┘                                                                             │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```