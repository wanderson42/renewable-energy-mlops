# Model Card — Wind Power Forecasting

## 1. Visão geral

Este documento descreve o modelo e o lifecycle de Machine Learning do projeto **Renewable Energy MLOps**.

O objetivo é prever geração eólica horária para operação **Day-Ahead** na Bahia. O modelo aprende o fator de capacidade (`target_fc`) e a previsão operacional é reconstruída em MW por:

```text
predicted_mw = predicted_fc × capacidade_mw
```

O projeto é educacional/de portfólio e não constitui um sistema certificado para decisões reais de despacho energético.

---

## 2. Identidade operacional

Estado operacional validado em **2026-10-01**:

| Campo | Valor |
|---|---|
| Registered Model operacional | `ensemble_lgb_xgb_rf_bahia` |
| Alias | `@champion` |
| Champion | v10 |
| Run ID | `d466ccb7fa294851a86a73b05d4e5735` |
| OOT MAE | 829.59 MW |
| OOT nMAE | 7.04% |
| OOT R² | 0.8477 |
| Número de features | 9 |

O último Challenger operacional anterior ao experimento v0.2.0 foi a **v12**:

| Métrica | Champion v10 | Challenger v12 |
|---|---:|---:|
| OOT MAE | 829.59 MW | 837.85 MW |
| OOT nMAE | 7.04% | 7.11% |
| OOT R² | 0.8477 | 0.8438 |
| Features | 9 | 9 |

A v12 foi rejeitada pelo Quality Gate e o alias `@champion` permaneceu na v10.

### Estado após o experimento v0.2.0

O experimento de simplificação realizado em 2026-10-01 **não alterou automaticamente o modelo operacional**.

Os resultados da v0.2.0 foram produzidos sobre um novo snapshot temporal congelado e são tratados como evidência experimental. Eles não são comparados diretamente com as métricas históricas da v10 para fins de promoção, pois os períodos OOT são diferentes.

---

## 3. Target

O target de treinamento é o fator de capacidade:

```text
target_fc = clip(wind_generation_mw / capacidade_mw, 0, 1)
```

A previsão em MW é reconstruída por:

```text
predicted_mw = predicted_fc × capacidade_mw
```

O erro operacional em MW é calculado após essa reconversão.

A série `capacidade_mw` utilizada no contrato atual é construída a partir de checkpoints históricos de capacidade instalada da **ABEEólica / INFOVENTO**, aplicados causalmente.

---

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

`wind_speed_roll_mean_3h` é causal e utiliza uma janela cronológica de três horas. A implementação não deve consultar observações futuras nem atravessar artificialmente grandes gaps temporais.

Modelos históricos podem possuir a mesma lista em ordem diferente. Para avaliação desses modelos, a entrada é alinhada à ordem esperada pelo estimator carregado sem alterar o contrato canônico atual.

---

## 5. Arquitetura

A arquitetura operacional atual é um Stacking temporal:

```text
LightGBM ───┐
XGBoost ────┼──> OOF causais ──> LinearRegression
RandomForest┘                    positive=True
                                  fit_intercept=False
```

A implementação `TemporalStackingRegressor` utiliza `TimeSeriesSplit` para gerar previsões OOF sem leakage futuro. O bloco inicial sem OOF é excluído do treinamento do meta-learner e os modelos base são refitados sobre todo o conjunto de treino ao final.

Na v0.2.0, a arquitetura passou a aceitar **estimadores base configuráveis**, permitindo executar ablações controladas sem criar trainers separados.

Arquiteturas avaliadas:

```text
FULL
lgbm + xgboost + rf

WITHOUT_XGB
lgbm + rf

WITHOUT_RF
lgbm + xgboost
```

Essa configurabilidade foi introduzida para suportar o experimento de simplificação e não implica mudança automática da arquitetura operacional.

---

## 6. Otimização

A otimização utiliza Optuna com validação temporal:

```text
Outer TimeSeriesSplit
    └── avaliação futura do trial

Inner TimeSeriesSplit
    └── OOF causal do meta-learner
```

No experimento v0.2.0:

```text
Optuna trials       = 20
Outer CV splits     = 3
Inner stacking CV   = 5
Features            = 9
```

O orquestrador aceita optimizer opcional, porém o trainer de Stacking atual exige:

```text
best_params
optimization_run_id
```

O espaço de busca respeita a arquitetura ativa: hiperparâmetros de estimadores desabilitados não fazem parte do estudo.

---

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

---

## 8. Experimento de simplificação — v0.2.0

O experimento foi executado sobre o mesmo snapshot, o mesmo TRAIN, o mesmo OOT, o mesmo contrato de 9 features e o mesmo orçamento de otimização para todas as arquiteturas.

### Snapshot experimental

```text
GOLD
rows: 22,145
range: 2024-03-21 00:00:00 → 2026-10-01 02:00:00
missing timestamps: 34
gaps: 2

TRAIN
rows: 21,425
range: 2024-03-21 00:00:00 → 2026-09-01 02:00:00

OOT — setembro/2026
rows: 720
range: 2026-09-01 03:00:00 → 2026-10-01 02:00:00
missing timestamps: 0
gaps: 0
```

### Resultados OOT

| Arquitetura | OOT MAE (MW) | OOT nMAE (%) | OOT R² |
|---|---:|---:|---:|
| LGBM + XGB + RF | 774.90 | 6.57 | 0.8666 |
| LGBM + RF | 788.35 | 6.69 | 0.8633 |
| LGBM + XGB | 785.31 | 6.66 | 0.8641 |

No ensemble completo, os coeficientes OOF do meta-learner foram:

```text
LGBM      0.3873
XGBoost   0.5356
RF        0.0439
```

Na arquitetura `LGBM + XGB`, os coeficientes foram:

```text
LGBM      0.4519
XGBoost   0.5151
```

A hipótese original de remover o XGBoost **não foi sustentada** neste snapshot. Sua remoção produziu a maior degradação entre as duas ablações.

A remoção do Random Forest produziu a melhor arquitetura reduzida, mas o ensemble completo ainda apresentou os melhores valores pontuais no OOT.

```text
Melhor desempenho OOT pontual:
LGBM + XGB + RF

Melhor candidato reduzido:
LGBM + XGB
```

Os coeficientes do meta-learner não são percentuais normalizados de importância e sua soma não é obrigada a ser igual a 1.

O experimento não estabelece equivalência estatística entre as arquiteturas e não realizou promoção automática de modelo.

---

## 9. Rastreabilidade no MLflow

A arquitetura ativa é registrada como parâmetro/tag no MLflow.

As Runs e Registered Models usam nomenclatura coerente com os estimadores habilitados:

```text
LGBM + XGB + RF
Run:   Stacking_LGB_XGB_RF_Bahia
Model: ensemble_lgb_xgb_rf_bahia

LGBM + RF
Run:   Stacking_LGB_RF_Bahia
Model: ensemble_lgb_rf_bahia

LGBM + XGB
Run:   Stacking_LGB_XGB_Bahia
Model: ensemble_lgb_xgb_bahia
```

O experimento de simplificação também é identificado por:

```text
experiment_family = ensemble_simplification_v0.2.0
```

Essa tag é específica do benchmark v0.2.0 e não representa um contrato permanente para futuros treinamentos.

---

## 10. Explainability

O treinamento gera artefatos SHAP:

```text
shap_summary.png
shap_dependence.png
shap_waterfall.png
shap_importance.csv
```

O serving/dashboard recupera esses artefatos usando o `run_id` da mesma Run efetivamente servida pela FastAPI.

A explicabilidade é usada como apoio diagnóstico. Importância SHAP ou pesos do meta-learner não são tratados isoladamente como critério suficiente para remoção de features ou estimadores.

---

## 11. Monitoring

O Evidently monitora exclusivamente as 9 features do contrato.

Regras atuais:

```text
Data Drift threshold        = 0.50
Performance Drift threshold = +2.0 p.p. de nMAE
CT trigger                  = Data Drift OR Performance Drift
```

Na execução E2E validada, 7 de 9 features apresentaram drift (77.8%), acionando Continuous Training.

Data Drift e Performance Drift são tratados como sinais distintos. Drift pode disparar retreinamento, mas não promove um modelo diretamente.

---

## 12. Quality Gate

O Monitoring pode disparar treinamento, mas não promoção.

A decisão Champion/Challenger considera atualmente:

- melhoria de MAE em MW; ou
- simplificação por menor número de **features**, desde que a degradação de nMAE permaneça dentro da tolerância configurada.

A tolerância atual de simplificação é **0.05 ponto percentual de nMAE**.

Essa regra de parcimônia **não representa simplificação por número de estimadores base**.

No experimento v0.2.0, todas as arquiteturas utilizam as mesmas 9 features. Portanto, a redução de três para dois modelos base não satisfaz, por si só, a condição de parcimônia do Quality Gate atual.

Uma futura política de simplificação arquitetural deve ser definida separadamente, somente se houver necessidade real de incorporá-la ao lifecycle operacional.

---

## 13. Dados e escopo

Fontes principais:

- **meteorologia:** Open-Meteo;
- **geração eólica horária:** dados abertos do ONS;
- **capacidade instalada canônica:** checkpoints históricos da ABEEólica / INFOVENTO;
- **ANEEL:** fonte diagnóstica independente para reconciliação de eventos de entrada em operação;
- **referência meteorológica do produto Day-Ahead:** Morro do Chapéu - BA.

Na extração do ONS, horas com registros de geração incompletos são descartadas antes da agregação estadual. Valores ausentes de usinas não são convertidos em `0 MW`.

A capacidade é aplicada como uma função degrau causal: em cada timestamp é utilizado somente o checkpoint mais recente já disponível. Não são utilizados interpolação linear nem backward fill.

Datasets reais não são versionados no Git. O Data Lake operacional reside no RustFS e os datasets de treino/OOT são registrados como lineage no MLflow.

O histórico canônico de capacidade da v0.2.0 começa em **2024-03-21**. Experimentos de treinamento que dependam de `capacidade_mw` não devem assumir dados canônicos anteriores a essa data.

---

## 14. Validação de software

No fechamento do experimento v0.2.0:

```text
pytest direcionado do trainer: 15 passed
tox -e py314:                  122 passed
```

A suíte cobre, entre outros pontos:

- contrato canônico de features;
- causalidade da rolling de 3 horas;
- capacidade temporal causal;
- auditoria do snapshot Gold;
- split temporal sem overlap;
- Temporal Stacking;
- otimização temporal;
- arquiteturas configuráveis;
- nomenclatura dinâmica de Runs e Registered Models;
- serving;
- monitoring;
- Quality Gate.

---

## 15. Limitações e riscos

- Métricas representam janelas temporais específicas e não garantem desempenho futuro.
- Mudanças de regime, frota, capacidade operacional e clima podem alterar a relevância de dados antigos.
- A relevância relativa dos estimadores pode mudar quando o snapshot ou a janela temporal muda.
- Pesos do meta-learner não constituem, isoladamente, evidência suficiente para remover um estimador.
- As diferenças entre o ensemble completo e `LGBM + XGB` não foram submetidas a um teste formal de equivalência ou não-inferioridade.
- Data Drift não implica necessariamente degradação de performance.
- A camada de compatibilidade histórica existe para governança entre gerações antigas e não deve virar contrato permanente para novas Runs.
- O ambiente validado é local/KinD e não representa requisitos de disponibilidade, segurança e compliance de produção energética real.

---

## 16. Próximos experimentos

Os próximos experimentos devem permanecer simples e responder a perguntas concretas.

Prioridades atuais:

- comparar a janela **expansiva** com janelas rolling de **12 e 24 meses**, utilizando o mesmo holdout futuro;
- reavaliar janelas mais longas somente quando houver histórico canônico suficiente de `capacidade_mw`;
- acompanhar a estabilidade de `LGBM + XGB + RF` e `LGBM + XGB` em novos meses OOT;
- medir custo/tempo de treinamento e inferência apenas se isso se tornar relevante para a decisão entre as duas arquiteturas;
- considerar no futuro uma identidade de Registered Model orientada ao produto, desacoplada da composição específica dos estimadores;
- manter **Cloud Infrastructure & IaC** como uma fase separada da evolução do modelo.
