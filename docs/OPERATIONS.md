# Operations Runbook

Este runbook descreve a operação local validada do projeto **Renewable Energy MLOps**.

## 1. Pré-requisitos

- Python 3.14
- Poetry 2.x
- Docker
- KinD
- kubectl
- Helm

O ambiente Python é reproduzido por `pyproject.toml` + `poetry.lock`.

```bash
poetry install
cp .env.example .env
```

Preencha o `.env` local com os valores do ambiente. Não versione secrets reais.

## 2. Componentes

No estado local validado:

| Componente | Porta local | Execução |
|---|---:|---|
| RustFS S3 API | 9000 | Kubernetes + port-forward |
| RustFS Console | 9001 | Kubernetes + port-forward |
| MLflow | 5000 | Kubernetes + port-forward |
| FastAPI | 8000 | Kubernetes + port-forward |
| Prefect UI | 4200 | processo local Poetry |
| Streamlit | 8501 | processo local Poetry |

PostgreSQL executa dentro do cluster e funciona como backend metadata store do MLflow.

## 3. Subir interfaces e port-forwards

Com o cluster e os workloads já provisionados:

```bash
make ports
```

Esse target inicia os port-forwards do RustFS, MLflow e FastAPI e também inicia Prefect Server e Streamlit localmente.

## 4. Diagnóstico

```bash
make status
```

`status` é observacional. Ele mostra:

- acesso ao cluster e contexto Kubernetes;
- readiness dos deployments;
- resposta HTTP dos serviços locais;
- Champion atualmente servido;
- digest/imageID da API em execução.

Para RustFS, uma resposta `403` na raiz sem autenticação é tratada como **reachable**, não como health check.

## 5. Gate operacional

```bash
make validate
```

`validate` falha com exit code diferente de zero quando algum requisito obrigatório não é atendido.

Semântica principal:

```text
RustFS API / Console → reachable
MLflow               → HTTP web service disponível
FastAPI /health      → exatamente HTTP 200
FastAPI /model-info  → exatamente HTTP 200
Prefect              → HTTP web service disponível
Dashboard            → HTTP web service disponível
```

A validação do ambiente complementa — e não substitui — `pytest/tox`.

## 6. Endpoints FastAPI

```text
GET  /health
GET  /model-info
POST /predict/batch
POST /reload-model
```

Exemplos:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model-info | python -m json.tool
```

## 7. Monitoring

Execução manual:

```bash
poetry run python -m energy_mlops.pipelines.monitoring_flow
```

O fluxo:

```text
Gold Reference + Gold Current
        ↓
9 canonical features
        ↓
Evidently Data Drift
        ↓
Performance Drift contra @champion
        ↓
Data Drift OR Performance Drift ?
        ├── não → encerra
        └── sim → Continuous Training
```

Thresholds atuais:

```text
Data Drift share       >= 0.50
Performance nMAE delta >= +2.0 p.p.
```

O relatório HTML é persistido antes da avaliação de Performance Drift.

## 8. Continuous Training

Execução direta do fluxo:

```bash
poetry run python -m energy_mlops.pipelines.training_flow
```

Fluxo padrão:

```text
janela temporal
    ↓
optimizer "stacking"
    ↓
Optuna
    ↓
TemporalStackingRegressor
    ↓
Challenger
    ↓
Quality Gate
    ↓
@champion ou REJECTED
```

O contrato também permite trainers independentes de optimizer:

```text
optimizer_name="none"
```

Nesse caso, o trainer recebe `best_params=None` e `optimization_run_id=None`.

## 9. Quality Gate e hot reload

Se um Challenger for promovido, o alias MLflow `@champion` é atualizado. Em seguida, o fluxo tenta:

```text
POST /reload-model
```

para recarregar o novo Champion sem reiniciar o processo da API.

Após qualquer mudança de Champion, valide:

```bash
curl -s http://localhost:8000/model-info | python -m json.tool
curl -s http://localhost:8000/health | python -m json.tool
```

Depois execute uma previsão Day-Ahead e confira o Dashboard/XAI.

## 10. Software lifecycle / CI

O GitHub Actions executa os testes e, se estiver verde, constrói e publica a imagem no GHCR.

O workflow **não executa rollout Kubernetes**.

Depois que uma nova imagem estiver disponível, o rollout local é explícito:

```bash
kubectl rollout restart deployment/energy-api
kubectl rollout status deployment/energy-api
```

Depois:

```bash
make status
make validate
```

A imagem realmente executada deve ser verificada pelo digest/imageID do pod, não apenas pela tag `latest`.

## 11. Testes

Suite isolada usada no CI/local:

```bash
poetry run tox -r -e py314
```

Validação consolidada em 2026-09-30:

```text
75 passed
0 failed
2 warnings
```

Checks de estilo/whitespace:

```bash
poetry run ruff check src tests
git diff --check
```

## 12. Encerrar serviços locais

```bash
make stop-ports
```

## 13. Troubleshooting

### RustFS retorna HTTP 403

Na raiz sem autenticação, `403` confirma que o serviço respondeu. Isso é suficiente para o teste de reachability utilizado pelo Makefile; não significa autenticação bem-sucedida.

### API responde 503 em `/health`

O readiness depende do Champion ter sido carregado na memória. Verifique MLflow, RustFS, credenciais e a existência do alias `@champion`.

### Dashboard/XAI não corresponde ao Champion

Confira `/model-info`. Os artefatos XAI devem ser recuperados pelo `run_id` efetivamente servido, e não pela “última Run” do experimento.

### Monitoring falha por feature order

Modelos históricos podem ter sido treinados com ordem diferente das mesmas features. A avaliação do Champion deve alinhar `X_current` à ordem esperada pelo estimator histórico sem alterar o contrato canônico do treinamento atual.

### Teste unitário tenta acessar MLflow/RustFS

Isso indica vazamento de dependência externa. A fronteira de carregamento deve ser mockada/isolada quando infraestrutura real não fizer parte do comportamento testado.

## 14. Evidências operacionais

Evidências E2E locais podem ser salvas para auditoria durante a validação, mas não precisam ser versionadas no Git. O notebook técnico consolida os resultados relevantes para o portfólio.
