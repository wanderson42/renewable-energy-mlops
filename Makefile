.PHONY: ports stop-ports test-opt test-train

ports:
	@echo "🚀 Iniciando redirecionamento de portas e serviços em segundo plano..."
	@nohup kubectl port-forward svc/rustfs 9000:9000 > .ports.log 2>&1 &
	@nohup kubectl port-forward svc/rustfs 9001:9001 >> .ports.log 2>&1 &
	@nohup kubectl port-forward svc/mlflow 5000:5000 >> .ports.log 2>&1 &
	@nohup kubectl port-forward svc/energy-api 8000:8000 >> .ports.log 2>&1 &
	@nohup poetry run prefect server start > prefect.log 2>&1 &
	@nohup poetry run streamlit run src/energy_mlops/app/app.py > dashboard.log 2>&1 &
	@sleep 3
	@echo "✅ Serviços liberados:"
	@echo "   - RustFS API:     http://localhost:9000"
	@echo "   - RustFS Console: http://localhost:9001"
	@echo "   - MLflow UI:     http://localhost:5000"
	@echo "   - FastAPI API:   http://localhost:8000"
	@echo "   - Prefect UI:    http://127.0.0.1:4200"
	@echo "   - Dashboard UI:  http://localhost:8501"
	@echo "   (Logs: .ports.log, prefect.log e dashboard.log)"

stop-ports:
	@echo "🛑 Encerrando serviços e redirecionamentos..."
	@-pkill -f "kubectl port-forward" || echo "Nenhum port-forward ativo."
	@-pkill -f "prefect server start" || echo "Nenhum Prefect ativo."
	@-pkill -f "streamlit run" || echo "Nenhum Streamlit ativo."
	@echo "✅ Portas e serviços fechados."

test-opt:
	@echo "🧪 Executando teste de otimização (Optuna -> MLflow)..."
	poetry run python -m energy_mlops.models.optimize --trials 5

test-train:
	@echo "🧪 Executando treino do Stacking Ensemble (MLflow Artifacts -> MinIO)..."
	poetry run python -m energy_mlops.models.train_ensemble