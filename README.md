O repositório foi estruturado seguindo os padrões modernos de engenharia de software e MLOps. 
A gestão de dependências é feita via Poetry e o pacote principal é instalável localmente (`energy_mlops`).

renewable-energy-mlops/
├── Dockerfile                          # Configuração da imagem do serviço FastAPI
├── Makefile                            # Automação (ports, stop-ports, test-opt, test-train)
├── pyproject.toml                      # Dependências (Poetry) e ferramentas de linting
├── poetry.lock                         # Lockfile de dependências exatas
├── README.md                   
├── lista_ordenada_de_comandos_bash.txt # Histórico de comandos e setup do projeto
├── helm/                               # Infraestrutura Kubernetes (KinD)
│   ├── Chart.yaml              
│   ├── values.yaml             
│   └── templates/              
│       ├── api.yaml                    # Deploy e Service do Model Serving (FastAPI)
│       ├── minio.yaml                  # Inclui PersistentVolumeClaim (PVC)
│       ├── mlflow.yaml                 # Deployment do MLflow Tracking Server
│       └── postgres.yaml               # Inclui PersistentVolumeClaim (PVC)
├── notebooks/                  
│   └── renewable-energy-mlops.ipynb    # Documentação e relatório técnico central
├── src/
│   └── energy_mlops/           
│       ├── __init__.py
│       ├── config.py                   # Pydantic Settings (Conexões S3, MLflow, etc.)
│       ├── data/               
│       │   ├── build_features.py       # Lógica pd.merge_asof para junção no tempo
│       │   ├── extract_energy.py
│       │   ├── extract_weather.py
│       │   └── schema.py               # Contratos de dados Pandera
│       ├── models/             
│       │   ├── optimize.py             # Otimização de hiperparâmetros (Optuna)
│       │   ├── train.py          
│       │   └── train_ensemble.py       # Treinamento do StackingRegressor
│       ├── pipelines/          
│       │   ├── backfill_flow.py        # Carga histórica no Prefect
│       │   ├── data_ingestion_flow.py  # Pipeline diário
│       │   └── utils.py
│       └── service/                    # Camada de Serviço (Model Serving)
│           ├── main.py                 # Endpoint assíncrono FastAPI (/predict/batch)
│           └── schema.py               # Contratos Pydantic para validação HTTP
└── tests/                      
    └── __init__.py



Diagrama arquitetural com o host, o gerenciamento de rede e o cluster Kubernetes com os Volumes Persistentes (PVCs):


     [ Cliente REST ]          ┌──────────────────────────────────────────────────────────────────────────────┐
            │                  │                         MÁQUINA HOST (Alienware-16)                          │
            │                  │                                                                              │
            │                  │     ┌───────────────────────┐             ┌─────────────────────────┐        │
            └─────────────────▶│     │ GERENCIAMENTO DE REDE │             │ AMBIENTE LOCAL (Poetry) │        │
       POST /predict/batch     │     │                       │             │                         │        │
     (http://localhost:8000)   │     │ ▹ make ports          │             │ ▹ Orquestração          │        │
                               │     │   (port-forward 8000) │             │ ▹ Model Training        │        │
                               │     └───────────┬───────────┘             └───────────┬─────────────┘        │
                               │                 │                                     │                      │
                               └─────────────────┼─────────────────────────────────────┼──────────────────────┘
                                                 │ Túnel K8s                           │ Conexões (9000, 5000)
                                                 ▼                                     ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   KUBERNETES CLUSTER (KinD - Docker Container)                                │
│                                   NODE: energy-mlops-control-plane                                            │
│                                                                                                               │
│             ┌─────────────────────────────┐      ┌──────────────────────────┐      ┌───────────────────────┐  │
│             │ POD: api-deployment-xxx     │      │ POD: mlflow--xxx         │      │ POD: postgres-xxx     │  │
│             │ (FastAPI Model Serving)     │      │ (Tracking Server)        │      │ (Database do MLflow)  │  │
│             │                             │      │                          │      │                       │  │
│             │ ┌─────────────────────────┐ │      │ ┌──────────────────────┐ │      │ ┌───────────────────┐ │  │
│             │ │ CONTAINER: energy-api   │─┼─────▶│ │ CONTAINER: mlflow    │─┼─────▶│ │ CONTAINER: post...│ │  │
│             │ │ ▹ :8000 (REST API)      │ │      │ │ ▹ :5000 (UI/API)     │ │      │ │ ▹ :5432 (TCP)     │ │  │
│             │ └─────────┬───────────────┘ │      │ └──────────────────────┘ │      │ └─────────┬─────────┘ │  │
│             └───────────┼─────────────────┘      └────────────┬─────────────┘      └───────────┼───────────┘  │
│                         │                                     │                                │  (PVC)       │
│                         │ Lê modelo @champion                 │ Salva Artefatos                ▼              │
│                         ▼                                     │                      [ postgres-pvc:2Gi ]     │
│             ┌─────────────────────────────┐                   │                                               │
│             │ POD: minio-xxx              │◀──────────────────┘                                               │
│             │ (Data Lake / Model Registry)│                                                                   │
│             │                             │                                                                   │
│             │ ┌─────────────────────────┐ │  (PVC)                                                            │
│             │ │ CONTAINER: minio        │─┼──────────▶ [ minio-pvc:5Gi ]                                      │
│             │ │ ▹ :9000 (S3 API)        │ │            (Garante retenção do Data Lake e Modelos)              │
│             │ └─────────────────────────┘ │                                                                   │
│             └─────────────────────────────┘                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
