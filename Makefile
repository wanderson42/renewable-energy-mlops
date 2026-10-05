SHELL := /bin/bash

.PHONY: ports stop-ports status validate repro-cluster
PORT_LOG_DIR := .ports

# ==============================================================================
# CONFIGURAÇÃO LOCAL
# ==============================================================================

API_URL := http://localhost:8000
MLFLOW_URL := http://localhost:5000
RUSTFS_API_URL := http://localhost:9000
RUSTFS_CONSOLE_URL := http://localhost:9001
PREFECT_URL := http://127.0.0.1:4200
DASHBOARD_URL := http://localhost:8501

K8S_DEPLOYMENTS := energy-api mlflow postgres rustfs


# ==============================================================================
# SERVIÇOS LOCAIS / PORT-FORWARD
# ==============================================================================

ports: stop-ports
	@echo "🚀 Iniciando redirecionamento de portas e serviços em segundo plano..."
	@mkdir -p $(PORT_LOG_DIR)
	@rm -f $(PORT_LOG_DIR)/*.log

	@nohup bash scripts/port-forward-supervisor.sh \
		rustfs-api rustfs 9000:9000 \
		> $(PORT_LOG_DIR)/rustfs-api.log 2>&1 &

	@nohup kubectl port-forward svc/rustfs 9001:9001 \
		> $(PORT_LOG_DIR)/rustfs-console.log 2>&1 &

	@nohup bash scripts/port-forward-supervisor.sh \
		mlflow mlflow 5000:5000 \
		> $(PORT_LOG_DIR)/mlflow.log 2>&1 &

	@nohup kubectl port-forward svc/energy-api 8000:8000 \
		> $(PORT_LOG_DIR)/energy-api.log 2>&1 &

	@nohup poetry run prefect server start \
		> prefect.log 2>&1 &

	@nohup poetry run streamlit run src/energy_mlops/app/app.py \
		> dashboard.log 2>&1 &

	@sleep 3

	@echo "✅ Serviços liberados:"
	@echo "   - RustFS API:     $(RUSTFS_API_URL)"
	@echo "   - RustFS Console: $(RUSTFS_CONSOLE_URL)"
	@echo "   - MLflow UI:      $(MLFLOW_URL)"
	@echo "   - FastAPI API:    $(API_URL)"
	@echo "   - Prefect UI:     $(PREFECT_URL)"
	@echo "   - Dashboard UI:   $(DASHBOARD_URL)"
	@echo "   - Logs de portas: $(PORT_LOG_DIR)/"


# Cria somente o KinD de ensaio, usando kubeconfig separado.
repro-cluster:
	@bash scripts/create-repro-cluster.sh

stop-ports:
	@echo "🛑 Encerrando serviços e redirecionamentos..."

	@-pkill -f "[p]ort-forward-supervisor.sh" \
		|| echo "Nenhum supervisor de port-forward ativo."

	@-pkill -f "[k]ubectl port-forward" \
		|| echo "Nenhum port-forward ativo."

	@-pkill -f "[p]refect server start" \
		|| echo "Nenhum Prefect ativo."

	@-pkill -f "[s]treamlit run" \
		|| echo "Nenhum Streamlit ativo."

	@sleep 1

	@echo "✅ Portas e serviços fechados."


# ==============================================================================
# STATUS DA INFRAESTRUTURA
# ==============================================================================

status:
	@echo "============================================================"
	@echo "📊 STATUS DA INFRAESTRUTURA MLOps"
	@echo "============================================================"
	@echo
	@echo "☸️  Kubernetes"
	@if kubectl cluster-info >/dev/null 2>&1; then \
		echo "   ✅ Cluster acessível"; \
		echo "   Contexto: $$(kubectl config current-context)"; \
	else \
		echo "   ❌ Cluster Kubernetes não acessível"; \
	fi
	@echo
	@for deployment in $(K8S_DEPLOYMENTS); do \
		if kubectl get deployment "$$deployment" >/dev/null 2>&1; then \
			ready=$$(kubectl get deployment "$$deployment" \
				-o jsonpath='{.status.readyReplicas}' 2>/dev/null); \
			desired=$$(kubectl get deployment "$$deployment" \
				-o jsonpath='{.status.replicas}' 2>/dev/null); \
			ready=$${ready:-0}; \
			desired=$${desired:-0}; \
			if [[ "$$ready" == "$$desired" && "$$desired" != "0" ]]; then \
				printf "   ✅ %-12s %s/%s Ready\n" \
					"$$deployment" "$$ready" "$$desired"; \
			else \
				printf "   ⚠️  %-12s %s/%s Ready\n" \
					"$$deployment" "$$ready" "$$desired"; \
			fi; \
		else \
			printf "   ❌ %-12s não encontrado\n" "$$deployment"; \
		fi; \
	done
	@echo
	@echo "🌐 Serviços locais"
	@check_http() { \
		name="$$1"; \
		url="$$2"; \
		code=$$(curl -sS \
			-o /dev/null \
			-w '%{http_code}' \
			--max-time 3 \
			"$$url" 2>/dev/null || true); \
		code=$${code:-000}; \
		if [[ "$$code" == "000" ]]; then \
			printf "   ❌ %-20s indisponível\n" "$$name"; \
		elif [[ "$$code" =~ ^5[0-9][0-9]$$ ]]; then \
			printf "   ⚠️  %-20s reachable (HTTP %s)\n" \
				"$$name" "$$code"; \
		else \
			printf "   ✅ %-20s reachable (HTTP %s)\n" \
				"$$name" "$$code"; \
		fi; \
	}; \
	check_http "RustFS API" "$(RUSTFS_API_URL)"; \
	check_http "RustFS Console" "$(RUSTFS_CONSOLE_URL)"; \
	check_http "MLflow" "$(MLFLOW_URL)"; \
	check_http "FastAPI" "$(API_URL)/health"; \
	check_http "Prefect" "$(PREFECT_URL)"; \
	check_http "Dashboard" "$(DASHBOARD_URL)"
	@echo
	@echo "🏆 Modelo em produção"
	@if curl -fsS \
		--max-time 3 \
		"$(API_URL)/model-info" 2>/dev/null \
		| poetry run python -c \
		'import json, sys; d=json.load(sys.stdin); print("   ✅ Champion v{} | run_id={}...".format(d["version"], d["run_id"][:8]))' \
		2>/dev/null; then \
		true; \
	else \
		echo "   ❌ Metadados do Champion indisponíveis"; \
	fi
	@echo
	@echo "🐳 Imagem da API"
	@image_id=$$(kubectl get pod \
		-l app=energy-api \
		-o jsonpath='{.items[0].status.containerStatuses[0].imageID}' \
		2>/dev/null || true); \
	if [[ -n "$$image_id" ]]; then \
		echo "   $$image_id"; \
	else \
		echo "   ⚠️  Image ID não disponível"; \
	fi
	@echo
	@echo "============================================================"


# ==============================================================================
# VALIDAÇÃO OPERACIONAL
# ==============================================================================

validate:
	@echo "============================================================"
	@echo "🔎 VALIDANDO INFRAESTRUTURA MLOps"
	@echo "============================================================"
	@echo
	@failures=0; \
	\
	echo "☸️  Kubernetes"; \
	if kubectl cluster-info >/dev/null 2>&1; then \
		echo "   ✅ Cluster acessível"; \
	else \
		echo "   ❌ Cluster Kubernetes não acessível"; \
		failures=$$((failures + 1)); \
	fi; \
	\
	for deployment in $(K8S_DEPLOYMENTS); do \
		if kubectl rollout status \
			"deployment/$$deployment" \
			--timeout=5s >/dev/null 2>&1; then \
			printf "   ✅ %-12s Ready\n" "$$deployment"; \
		else \
			printf "   ❌ %-12s não está Ready\n" "$$deployment"; \
			failures=$$((failures + 1)); \
		fi; \
	done; \
	\
	echo; \
	echo "🌐 Serviços locais"; \
	\
	require_exact_http() { \
		name="$$1"; \
		url="$$2"; \
		expected="$$3"; \
		code=$$(curl -sS \
			-o /dev/null \
			-w '%{http_code}' \
			--max-time 3 \
			"$$url" 2>/dev/null || true); \
		code=$${code:-000}; \
		if [[ "$$code" == "$$expected" ]]; then \
			printf "   ✅ %-20s healthy (HTTP %s)\n" \
				"$$name" "$$code"; \
		else \
			printf "   ❌ %-20s HTTP %s (esperado %s)\n" \
				"$$name" "$$code" "$$expected"; \
			failures=$$((failures + 1)); \
		fi; \
	}; \
	\
	require_web_service() { \
		name="$$1"; \
		url="$$2"; \
		code=$$(curl -sS \
			-o /dev/null \
			-w '%{http_code}' \
			--max-time 3 \
			"$$url" 2>/dev/null || true); \
		code=$${code:-000}; \
		if [[ "$$code" =~ ^[23][0-9][0-9]$$ ]]; then \
			printf "   ✅ %-20s reachable (HTTP %s)\n" \
				"$$name" "$$code"; \
		else \
			printf "   ❌ %-20s HTTP %s\n" \
				"$$name" "$$code"; \
			failures=$$((failures + 1)); \
		fi; \
	}; \
	\
	require_reachable() { \
		name="$$1"; \
		url="$$2"; \
		code=$$(curl -sS \
			-o /dev/null \
			-w '%{http_code}' \
			--max-time 3 \
			"$$url" 2>/dev/null || true); \
		code=$${code:-000}; \
		if [[ "$$code" =~ ^[1-4][0-9][0-9]$$ ]]; then \
			printf "   ✅ %-20s reachable (HTTP %s)\n" \
				"$$name" "$$code"; \
		else \
			printf "   ❌ %-20s HTTP %s\n" \
				"$$name" "$$code"; \
			failures=$$((failures + 1)); \
		fi; \
	}; \
	\
	require_reachable \
		"RustFS API" \
		"$(RUSTFS_API_URL)"; \
	\
	require_reachable \
		"RustFS Console" \
		"$(RUSTFS_CONSOLE_URL)"; \
	\
	require_web_service \
		"MLflow" \
		"$(MLFLOW_URL)"; \
	\
	require_exact_http \
		"FastAPI /health" \
		"$(API_URL)/health" \
		"200"; \
	\
	require_exact_http \
		"FastAPI /model-info" \
		"$(API_URL)/model-info" \
		"200"; \
	\
	require_web_service \
		"Prefect" \
		"$(PREFECT_URL)"; \
	\
	require_web_service \
		"Dashboard" \
		"$(DASHBOARD_URL)"; \
	\
	echo; \
	echo "============================================================"; \
	if [[ "$$failures" -eq 0 ]]; then \
		echo "✅ Infraestrutura validada com sucesso."; \
		echo "============================================================"; \
		exit 0; \
	else \
		echo "❌ Validação concluída com $$failures falha(s)."; \
		echo "   Verifique os serviços acima."; \
		echo "   Se necessário, execute: make ports"; \
		echo "============================================================"; \
		exit 1; \
	fi
