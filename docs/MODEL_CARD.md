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

### Estado após os experimentos posteriores

Os experimentos de simplificação arquitetural e de comparação de janelas temporais **não alteraram automaticamente o modelo operacional**.

Os resultados experimentais foram produzidos sobre snapshots e protocolos temporais explicitamente congelados e são tratados como evidência para evolução do sistema.

Eles não substituem automaticamente as métricas históricas do Champion nem implicam promoção sem passar pelo lifecycle operacional e pelo Quality Gate.

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

Essa configurabilidade foi introduzida para suportar experimentos controlados e não implica mudança automática da arquitetura operacional.

---

## 6. Otimização

A otimização utiliza Optuna com validação temporal:

```text
Outer TimeSeriesSplit
    └── avaliação futura do trial

Inner TimeSeriesSplit
    └── OOF causal do meta-learner
```

Nos benchmarks controlados recentes:

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

A janela de treinamento também pode ser parametrizada independentemente da arquitetura por meio de `training_window_months`.

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

## 8.1. Experimento de janela temporal de treinamento

Após o experimento de simplificação da arquitetura, foi realizado um segundo benchmark controlado para avaliar se restringir o histórico recente poderia melhorar a generalização temporal.

A arquitetura foi mantida fixa em:

```text
LGBM + XGBoost + Random Forest
```

Também permaneceram constantes:

```text
Features             = 9
OOT                  = setembro/2026
Optuna trials        = 20
Outer CV splits      = 3
Inner stacking CV    = 5
```

A única variável experimental foi a janela utilizada na construção do TRAIN:

```text
Expanding
Rolling 24 meses
Rolling 12 meses
```

O benchmark foi executado em uma rodada própria de otimização e treinamento. Portanto, os valores da configuração Expanding desta seção pertencem a esse benchmark e não substituem os resultados pontuais do experimento de simplificação da seção anterior.

### Janelas avaliadas

```text
EXPANDING
rows: 21,425
range: 2024-03-21 00:00:00 → 2026-09-01 02:00:00

ROLLING 24m
rows: 17,486
range: 2024-09-01 03:00:00 → 2026-09-01 02:00:00

ROLLING 12m
rows: 8,760
range: 2025-09-01 03:00:00 → 2026-09-01 02:00:00
```

O OOT permaneceu congelado:

```text
OOT — setembro/2026
rows: 720
range: 2026-09-01 03:00:00 → 2026-10-01 02:00:00
```

### Resultados

| Janela | TRAIN rows | CV nMAE mean (%) | CV nMAE std (%) | OOT MAE (MW) | OOT nMAE (%) | OOT R² |
|---|---:|---:|---:|---:|---:|---:|
| Expanding | 21,425 | 8.6949 | 0.6791 | 787.18 | 6.6788 | 0.8614 |
| Rolling 24m | 17,486 | 9.1920 | 1.6572 | 937.14 | 7.9511 | 0.8217 |
| Rolling 12m | 8,760 | 8.1086 | 0.8133 | 889.66 | 7.5483 | 0.8297 |

Para este snapshot, esta arquitetura e este holdout futuro, a **Expanding Window** apresentou o melhor desempenho OOT.

Em relação à Expanding:

```text
Rolling 24m
Δ MAE   ≈ +149.95 MW
Δ nMAE  ≈ +1.27 p.p.

Rolling 12m
Δ MAE   ≈ +102.48 MW
Δ nMAE  ≈ +0.87 p.p.
```

O Rolling 12m apresentou o menor nMAE médio na validação temporal interna, mas essa vantagem não foi reproduzida no OOT.

Esse resultado reforça que desempenho em CV temporal e desempenho em um holdout futuro devem ser analisados separadamente.

A diferença observada não demonstra que a Expanding Window seja universalmente superior. A conclusão permanece restrita ao snapshot, período OOT, arquitetura e protocolo experimental utilizados.

### Sensibilidade do meta-learner à janela

Os coeficientes OOF aprendidos também mudaram entre as janelas:

```text
Expanding
LGBM      0.5890
XGBoost   0.3809
RF        0.0000

Rolling 24m
LGBM      0.0769
XGBoost   0.2916
RF        0.5692

Rolling 12m
LGBM      0.5177
XGBoost   0.4331
RF        0.0000
```

A contribuição relativa dos estimadores, portanto, mostrou sensibilidade à janela temporal utilizada.

O Random Forest recebeu coeficiente zero na Expanding e no Rolling 12m, mas assumiu o maior coeficiente no Rolling 24m.

Por essa razão, os três modelos base foram mantidos fixos durante todo o benchmark. Dessa forma, a janela de treinamento permaneceu como a única variável experimental principal.

### Contrato implementado

A política de treinamento passou a aceitar:

```text
training_window_months=None  -> Expanding
training_window_months=24    -> Rolling 24m
training_window_months=12    -> Rolling 12m
```

A configuração Expanding permanece como referência experimental atual.

Nenhuma alteração automática do Champion foi realizada em decorrência desse benchmark.

A evidência completa está registrada em:

```text
notebooks/experiments/training_window_comparison_benchmark.ipynb
```

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

Os experimentos de janela temporal mantêm rastreabilidade própria de suas Runs de otimização e treinamento, preservando a separação entre evidência experimental e estado operacional do Champion.

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

A escolha da janela temporal também não constitui atualmente um critério automático de promoção. Ela é uma política de construção do dataset de treino que deve continuar sendo avaliada experimentalmente.

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

O histórico canônico de capacidade começa em **2024-03-21**. Experimentos de treinamento que dependam de `capacidade_mw` não devem assumir dados canônicos anteriores a essa data.

---

## 14. Validação de software

No fechamento da etapa de comparação de janelas e resiliência local:

```text
tox -e py314: 129 passed
              0 failed
              2 warnings
```

Os warnings conhecidos permanecem associados à compatibilidade interna Evidently/NumPy e não representam falhas da aplicação.

A suíte cobre, entre outros pontos:

- contrato canônico de features;
- causalidade da rolling de 3 horas;
- capacidade temporal causal;
- auditoria do snapshot Gold;
- split temporal sem overlap;
- seleção causal da janela de treinamento;
- compatibilidade entre Expanding e Rolling windows;
- propagação de `training_window_months` pelo training flow;
- Temporal Stacking;
- otimização temporal;
- arquiteturas configuráveis;
- nomenclatura dinâmica de Runs e Registered Models;
- serving;
- monitoring;
- Quality Gate;
- comportamento de retry do supervisor de port-forward;
- validação dos argumentos do supervisor;
- encerramento seguro do processo `kubectl` filho.

Testes destrutivos ou dependentes de uma infraestrutura KinD real não fazem parte da suíte unitária executada pelo Tox.

Reinícios reais de PostgreSQL, RustFS e do control-plane KinD foram utilizados como validação operacional manual e permanecem documentados separadamente em:

```text
docs/INFRASTRUCTURE.md
```

---

## 15. Limitações e riscos

- Métricas representam janelas temporais específicas e não garantem desempenho futuro.
- Mudanças de regime, frota, capacidade operacional e clima podem alterar a relevância de dados antigos.
- A vantagem observada da Expanding Window foi demonstrada em um único OOT congelado e deve ser reavaliada longitudinalmente.
- O melhor CV temporal interno não necessariamente corresponde ao melhor desempenho em um holdout futuro.
- A relevância relativa dos estimadores pode mudar quando o snapshot ou a janela temporal muda.
- Pesos do meta-learner não constituem, isoladamente, evidência suficiente para remover um estimador.
- As diferenças entre o ensemble completo e `LGBM + XGB` não foram submetidas a um teste formal de equivalência ou não-inferioridade.
- Os experimentos de janela temporal também não estabelecem superioridade estatística universal de uma política de treinamento.
- Data Drift não implica necessariamente degradação de performance.
- A camada de compatibilidade histórica existe para governança entre gerações antigas e não deve virar contrato permanente para novas Runs.
- O ambiente validado é local/KinD e não representa requisitos de disponibilidade, segurança e compliance de produção energética real.
- A persistência atualmente validada cobre reinícios de pods, deployments e do container control-plane do KinD, mas não demonstra durabilidade após deleção de PVC, destruição do cluster ou perda do host.

---

## 16. Próximos experimentos

Os próximos experimentos devem permanecer simples e responder a perguntas concretas, isolando sempre que possível uma variável principal por vez.

Prioridades atuais:

- repetir o benchmark de janelas com novos meses OOT fechados para avaliar se a vantagem observada da **Expanding Window** permanece estável ao longo do tempo;

- reavaliar janelas mais longas somente quando houver histórico canônico suficiente de `capacidade_mw`;

- acompanhar longitudinalmente a estabilidade de `LGBM + XGB + RF` e `LGBM + XGB` em novos períodos OOT;

- definir uma política explícita de simplificação por número de estimadores somente se houver necessidade de incorporá-la ao lifecycle operacional, mantendo-a separada da regra atual de parcimônia por número de features;

- medir custo de treinamento, latência de inferência e tamanho dos artefatos caso esses fatores passem a ser relevantes para a escolha entre arquiteturas;

- considerar futuramente uma identidade de Registered Model orientada ao produto, desacoplada da composição específica dos estimadores;

- remover a compatibilidade de métricas `*_YYYY` quando nenhum modelo operacional relevante depender mais do contrato histórico;

- manter **Cloud Infrastructure & IaC** como uma fase separada da evolução do modelo.

A estratégia de backup/restore para cenários além da fronteira de persistência atualmente validada — como recriação completa do cluster, deleção de PVC ou perda do host — pertence à evolução da infraestrutura e é acompanhada em `docs/INFRASTRUCTURE.md`, não como experimento de modelagem.
