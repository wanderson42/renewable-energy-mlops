# Infrastructure Architecture and Resilience

## Scope

This document describes the local MLOps infrastructure used by the
`renewable-energy-mlops` project, with emphasis on:

- configuration contracts;
- MLflow metadata and artifact storage;
- Kubernetes persistence;
- local port-forwarding;
- resilience behavior validated experimentally.

The environment documented here is the local KinD + Helm stack. It is
not a production-cloud architecture.

### Reproducible deployment milestone — 2026-10-05

The v1.0 rehearsal adds an isolated KinD cluster and a Terraform-managed Helm
release, with pinned node/workload images, verified PostgreSQL/S3 snapshot restore,
paired v17 inference, and declarative recovery from a controlled API image-pull
failure. The packaged MLflow runtime was adopted by digest and validated against
the live Registry, one model artifact and preserved PVC identities.

Prefect and Streamlit remain Poetry host processes, with dedicated loopback ports
and a separate Prefect profile/SQLite database. The operator confirmed dashboard
operation and that the Prefect UI error disappeared after the auth fix. No flow
execution or orchestration-history restoration is claimed by that UI observation.
RustFS still requires S3 credentials; the local Prefect server's unauthenticated
loopback access is not a cloud access-control design.

The original v0.3 cluster continues longitudinal monitoring independently. Its
manual rollout and port-forward supervisors are distinct from the Terraform/Helm
rehearsal procedure. AWS infrastructure remains a planned blueprint, with no cloud
resources deployed. Commands, exact identities, evidence and limits are in
[`REPRODUCIBLE_DEPLOYMENT.md`](REPRODUCIBLE_DEPLOYMENT.md).

---

## 1. Architecture overview

```text
Local Python / Notebook / CLI
        |
        | localhost:5000
        v
MLflow Tracking Server
        |
        +--------------------------+
        |                          |
        | metadata                 | artifacts
        v                          v
PostgreSQL                    RustFS (S3-compatible)
service: postgres:5432        service: rustfs:9000
        |                          |
        v                          v
postgres-pvc                  rustfs-pvc
/var/lib/postgresql/data      /data
```

The MLflow server uses:

```text
Backend Store:
postgresql://...@postgres:5432/...

Artifact Root:
s3://mlflow-artifacts/

S3 Endpoint inside Kubernetes:
http://rustfs:9000
```

The application does not rely on the MLflow container filesystem for
durable experiment state.

---

## 2. Configuration contract

### 2.1 Project-facing variables

`.env.example` is the public configuration contract for local
development. Real credentials remain in the ignored `.env`.

Relevant variables:

```text
RUSTFS_ENDPOINT
RUSTFS_BUCKET
RUSTFS_ROOT_USER
RUSTFS_ROOT_PASSWORD

MLFLOW_TRACKING_URI

POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_HOST
POSTGRES_PORT
```

In the Helm/Kubernetes environment, database and storage credentials are
managed separately through the local ignored Helm secrets values file.

### 2.2 RustFS -> boto3 / MLflow adapter

The project uses `RUSTFS_*` names as its own configuration vocabulary.
However, local MLflow artifact operations use boto3 internally, which
expects AWS/S3-compatible environment variables.

For local commands that directly upload or download MLflow artifacts,
the adapter is:

```text
RUSTFS_ROOT_USER
    -> AWS_ACCESS_KEY_ID

RUSTFS_ROOT_PASSWORD
    -> AWS_SECRET_ACCESS_KEY

RUSTFS_ENDPOINT
    -> MLFLOW_S3_ENDPOINT_URL
```

Example for the current shell session:

```bash
set -a
source .env
set +a

export AWS_ACCESS_KEY_ID="$RUSTFS_ROOT_USER"
export AWS_SECRET_ACCESS_KEY="$RUSTFS_ROOT_PASSWORD"
export MLFLOW_S3_ENDPOINT_URL="$RUSTFS_ENDPOINT"
```

This avoids duplicating the same credentials in `.env.example` under
two different naming schemes.

### 2.3 Local versus in-cluster endpoints

The endpoint depends on where the client is running:

```text
Inside Kubernetes:
http://rustfs:9000

From the local host through kubectl port-forward:
http://localhost:9000
```

Likewise, the local MLflow tracking endpoint is:

```text
http://localhost:5000
```

---

## 3. Persistent storage

### PostgreSQL

```text
PVC: postgres-pvc
Mount: /var/lib/postgresql/data
Capacity: 2 GiB
Access mode: RWO
```

PostgreSQL stores the MLflow backend metadata, including experiments,
runs, parameters, metrics, tags, and Model Registry state.

### RustFS

```text
PVC: rustfs-pvc
Mount: /data
Capacity: 5 GiB
Access mode: RWO
```

RustFS stores S3-compatible objects, including the `energy-lake` and
`mlflow-artifacts` buckets.

The RustFS process was verified to start with `/data` as its data
directory.

### Reclaim policy

The current PVs use:

```text
ReclaimPolicy: Delete
```

This is sufficient for pod and node/container restarts while the PVCs
remain present.

It must **not** be interpreted as a guarantee that data will survive
PVC deletion or `kind delete cluster`. Cluster destruction is a
different persistence boundary and has not been validated by the tests
documented here.

---

## 4. Local port-forward resilience

Long-running local training and experiment tracking depend on two
critical forwarded endpoints:

```text
localhost:5000 -> svc/mlflow:5000
localhost:9000 -> svc/rustfs:9000
```

A plain background command such as:

```bash
nohup kubectl port-forward ... &
```

survives terminal closure but does not restart itself after a broken
connection.

The project therefore uses:

```text
scripts/port-forward-supervisor.sh
```

for the two critical endpoints.

Current behavior:

```text
MLflow 5000
kubectl exits
    -> supervisor waits 2 seconds
    -> new kubectl port-forward is created

RustFS API 9000
kubectl exits
    -> supervisor waits 2 seconds
    -> new kubectl port-forward is created
```

The RustFS console (`9001`) and FastAPI (`8000`) remain simple
port-forwards because they are not part of the critical local training
and MLflow artifact path.

`make ports` first invokes `stop-ports`, preventing duplicate listeners
from accumulating during normal use.

Logs are separated under:

```text
.ports/
├── mlflow.log
├── rustfs-api.log
├── rustfs-console.log
└── energy-api.log
```

---

## 5. Failure discovered during the training-window benchmark

The training-window benchmark exposed a local infrastructure weakness:
a long Optuna/MLflow workflow could complete expensive computation and
then fail when the MLflow `kubectl port-forward` process disappeared.

The failure was not caused by loss of MLflow metadata or artifacts.

The investigation separated the concerns:

```text
Persistence:
PostgreSQL PVC -> healthy
RustFS PVC     -> healthy

Availability:
local kubectl port-forward -> fragile
```

The remediation was intentionally narrow: add automatic recovery only
to the critical local forwards required by experiment tracking and
artifact access.

---

## 6. Validation evidence

The following scenarios were executed manually against the local KinD
environment.

| Scenario | Expected behavior | Result |
|---|---|---|
| Kill MLflow `kubectl port-forward` | Supervisor creates a new process; `/health` returns | PASS |
| Kill RustFS API `kubectl port-forward` | Supervisor creates a new process; MLflow artifact remains downloadable | PASS |
| Restart PostgreSQL deployment | MLflow run metadata remains available | PASS |
| Restart RustFS deployment | Artifact remains stored in RustFS PVC and downloadable after connectivity returns | PASS |
| Restart KinD control-plane container | Workloads restart, critical forwards recover, metadata and artifact remain available | PASS |

The persistence canary validated both stores independently:

```text
Metadata:
parameter = survives_restart
metric    = 123.456

Artifact:
persistence_canary_v2.txt
```

After the KinD control-plane restart, the same run metadata and the same
artifact were recovered successfully.

---

## 7. What these tests prove

The validated recovery boundary is:

```text
individual pod restart
        +
deployment rollout restart
        +
KinD control-plane container restart
```

Within that boundary:

```text
MLflow metadata persists        -> PostgreSQL PVC
MLflow artifacts persist        -> RustFS PVC
MLflow local access recovers    -> port-forward supervisor
RustFS local access recovers    -> port-forward supervisor
```

The tests do **not** prove durability across:

```text
PVC deletion
kind delete cluster
host disk loss
machine loss
```

Those scenarios require a separate backup / external persistence
strategy and belong to a future infrastructure phase.

---

## 8. Operational diagnostics

Useful checks:

```bash
make status

pgrep -af 'port-forward-supervisor.sh'
pgrep -af 'kubectl port-forward'

ss -ltnp | grep -E ':5000|:9000'

curl -fsS http://127.0.0.1:5000/health
```

Supervisor logs:

```bash
tail -n 50 .ports/mlflow.log
tail -n 50 .ports/rustfs-api.log
```

Persistent storage:

```bash
kubectl get pvc -o wide
kubectl get pv -o wide
```

---

## 9. Documentation boundaries

The project documentation follows these roles:

```text
README.md
    -> project landing page and quick start

notebooks/renewable-energy-mlops.ipynb
    -> curated technical narrative, important experiments,
       architectural decisions, E2E lessons

notebooks/experiments/*.ipynb
    -> detailed controlled experiments and frozen evidence

docs/OPERATIONS.md
    -> operational runbook and commands

docs/INFRASTRUCTURE.md
    -> architecture, configuration, persistence and resilience

docs/MODEL_CARD.md
    -> model contract, metrics, limitations and governance
```

A development branch does not automatically require a notebook. A
dedicated notebook is appropriate when the work is itself an
experiment or analytical investigation. Infrastructure and operational
changes are normally documented in Markdown and summarized in the main
technical notebook only when they materially change the project's
technical story.
