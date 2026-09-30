# Model Card — Wind Power Forecasting

## 1. Visão geral

Este documento descreve o modelo e o lifecycle de Machine Learning do projeto **Renewable Energy MLOps**.

O objetivo é prever geração eólica horária para operação **Day-Ahead** na Bahia. O modelo aprende o fator de capacidade (`target_fc`) e a previsão operacional é reconstruída em MW por:

```text
predicted_mw = predicted_fc × capacidade_mw
```

O projeto é educacional/de portfólio e não constitui um sistema certificado para decisões reais de despacho energético.

## 2. Identidade operacional

Snapshot validado em **2026-09-30**:

| Campo | Valor |
|---|---|
| Registered Model | `ensemble_lgb_xgb_rf_bahia` |
| Alias | `@champion` |
| Champion | v10 |
| Run ID | `d466ccb7fa294851a86a73b05d4e5735` |
| OOT MAE | 829.59 MW |
| OOT nMAE | 7.04% |
| OOT R² | 0.8477 |
| Número de features | 9 |

O último Challenger real validado foi a **v12**:

| Métrica | Champion v10 | Challenger v12 |
|---|---:|---:|
| OOT MAE | 829.59 MW | 837.85 MW |
| OOT nMAE | 7.04% | 7.11% |
| OOT R² | 0.8477 | 0.8438 |
| Features | 9 | 9 |

A v12 foi rejeitada pelo Quality Gate; o alias `@champion` permaneceu na v10.

## 3. Target

O target de treinamento é o fator de capacidade:

```text
target_fc = clip(wind_generation_mw / capacidade_mw, 0, 1)
```

O erro operacional em MW é calculado somente após reconverter as previsões de fator de capacidade pela capacidade correspondente.

## 4. Contrato de features

A fonte única do contrato é `ModelFeatureSchema`.

Ordem canônica:

```text
1. temperature_2m
2. wind_speed_100m
3. wind_direction_100m
4. wind_temp_ratio
5. hour_sin
6. hour_cos
7. month_sin
8. month_cos
9. wind_speed_roll_mean_3h
```

`wind_speed_roll_mean_3h` é causal: o cálculo não deve consultar observações futuras.

Modelos históricos podem possuir a mesma lista em ordem diferente. Para avaliação desses modelos, a entrada é alinhada à ordem esperada pelo estimator carregado sem alterar o contrato canônico atual.

## 5. Arquitetura

O modelo atual é um Stacking temporal:

```text
LightGBM ───┐
XGBoost ────┼──> OOF causais ──> LinearRegression
RandomForest┘                    positive=True
                                  fit_intercept=False
```

A implementação `TemporalStackingRegressor` utiliza `TimeSeriesSplit` para gerar OOF sem leakage futuro. O bloco inicial sem OOF é excluído do treinamento do meta-learner e os modelos base são refitados sobre todo o conjunto de treino ao final.

## 6. Otimização

A otimização utiliza Optuna com validação temporal:

```text
Outer TimeSeriesSplit
    └── avaliação futura do trial

Inner TimeSeriesSplit
    └── OOF causal do meta-learner
```

O orquestrador aceita optimizer opcional, porém o trainer de Stacking atual exige:

```text
best_params
optimization_run_id
```

Isso preserva a separação entre o contrato geral `ModelTrainer` e as necessidades específicas de cada algoritmo.

## 7. Métricas

Novas Runs registram nomes canônicos:

```text
oot_mae_mw
oot_nmae_pct
oot_mae_fc_pct
oot_r2_score
```

O projeto não usa `100 - nMAE` como “accuracy”. nMAE permanece uma métrica de erro normalizado.

Uma camada temporária de compatibilidade aceita uma única variante histórica `_<YYYY>` quando a métrica canônica não existe. Múltiplas variantes anuais são tratadas como ambiguidade e geram erro.

## 8. Explainability

O treinamento gera artefatos SHAP:

```text
shap_summary.png
shap_dependence.png
shap_waterfall.png
shap_importance.csv
```

O serving/dashboard recupera esses artefatos usando o `run_id` da mesma Run efetivamente servida pela FastAPI.

## 9. Monitoring

O Evidently monitora exclusivamente as 9 features do contrato.

Regras atuais:

```text
Data Drift threshold        = 0.50
Performance Drift threshold = +2.0 p.p. de nMAE
CT trigger                  = Data Drift OR Performance Drift
```

Na execução E2E validada, 7 de 9 features apresentaram drift (77.8%), acionando Continuous Training.

## 10. Quality Gate

O Monitoring pode disparar treinamento, mas não promoção.

A decisão Champion/Challenger considera:

- melhoria de MAE em MW; ou
- simplificação por menor número de features, desde que a degradação de nMAE permaneça dentro da tolerância configurada.

A tolerância atual de simplificação é **0.05 ponto percentual de nMAE**.

## 11. Dados e escopo

Fontes principais:

- meteorologia: Open-Meteo;
- geração e capacidade: ONS;
- localização operacional do Dashboard: Morro do Chapéu - BA.

Datasets reais não são versionados no Git. O Data Lake operacional reside no RustFS e os datasets de treino/OOT são registrados como lineage no MLflow.

## 12. Limitações e riscos

- Métricas representam janelas históricas específicas e não garantem desempenho futuro.
- Mudanças de regime, frota, capacidade operacional e clima podem alterar a relevância de dados antigos.
- Data Drift não implica necessariamente degradação de performance; as duas dimensões são avaliadas separadamente.
- O último Challenger atribuiu peso zero ao XGBoost, mas isso não demonstra inutilidade geral do algoritmo.
- A camada de compatibilidade histórica existe para permitir governança entre gerações; não deve virar contrato permanente para novas Runs.
- O ambiente validado é local/KinD e não representa requisitos de disponibilidade, segurança e compliance de produção energética real.

## 13. Próximos experimentos

- rolling windows de 12 / 24 / 36 meses versus janela expansiva;
- `LightGBM + RandomForest` versus stack completo;
- estabilidade das métricas ao longo de múltiplos meses futuros;
- evolução do nome do Registered Model para uma identidade orientada ao produto.
