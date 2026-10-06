# Previsão de geração eólica na Bahia

**Resumo executivo para stakeholders — Renewable Energy MLOps**

**Autor:** Wanderson Ferreira · **Data:** 06/10/2026

> Previsão horária com rastreabilidade e monitoramento. A entrega v1.0 integra modelo em operação local, reprodução em Kubernetes e infraestrutura AWS como código, formando uma base para um piloto de produção supervisionado.

### 1. Problema e decisão apoiada

A variabilidade do vento dificulta antecipar a geração. O projeto estima a geração eólica horária agregada da Bahia e apresenta uma visão das próximas 24 horas. Seu uso proposto é apoiar a análise de horários de maior e menor geração e a investigação de desvios entre previsão e observação.

O projeto transforma dados meteorológicos e energéticos em previsões acessíveis, com identidade do modelo e acompanhamento de qualidade. Seu mérito está na integração entre modelagem e operação: o portfólio entrega uma solução executável, com uma trajetória definida para avaliar benefícios de planejamento em um piloto.

### 2. Resultado histórico e sua interpretação

A versão operacional é a v17. Na avaliação temporal de setembro de 2026, posterior ao treinamento, foram registrados:

| Indicador | Resultado | Como interpretar |
| --- | --- | --- |
| Erro absoluto médio (MAE) | 787,18 MW | Magnitude média do erro de previsão, expressa na unidade física da geração. |
| Erro normalizado (nMAE) | 6,68% | Erro médio equivalente a 6,68% da capacidade instalada máxima do período avaliado. |
| Coeficiente de determinação (R²; fator de capacidade) | 0,8614 | Soma dos erros quadráticos 86,14% menor que a referência constante na média do período. |
| Comparação com a v10 | 803,37 → 787,18 MW | Redução de aproximadamente 2% no erro médio em MW, na mesma janela de avaliação. |

O R² indica um ajuste histórico promissor e complementa a leitura do erro em MW [1]. Ele é calculado sobre o fator de capacidade; MAE e nMAE usam MW. R² e nMAE expressam ajuste e erro normalizado, respectivamente, e devem ser interpretados segundo essas definições.

Esses resultados usam meteorologia observada/reanálise. O próximo marco é medir o desempenho day-ahead com previsões meteorológicas arquivadas na emissão, comparadas depois à geração observada [2].

### 3. Por que Bahia e Morro do Chapéu?

A relevância do setor justifica o recorte baiano: segundo a SDE, a Bahia liderou a geração eólica nacional, com aproximadamente 37% do total em 2025 [3]. Morro do Chapéu é um polo com empreendimentos reais: a Enel anunciou em 2021 o início da operação de Morro do Chapéu Sul II (353 MW), próximo ao Sul (172 MW), em operação desde 2018 [4].

O projeto combina meteorologia Open-Meteo, geração ONS e capacidade ABEEólica/INFOVENTO. Morro do Chapéu oferece uma referência inicial relevante; avaliar sua representatividade espacial e comparar múltiplos pontos permitirá aperfeiçoar a previsão estadual.

## Como ler o produto

[Ver a captura do dashboard](https://github.com/wanderson42/renewable-energy-mlops/blob/8ffeed81cde9a7f690f50f9777afa00fd152aa97/notebooks/streamlit_day_ahead.png)

Recorte da captura atual do dashboard, publicada na main. Os valores ilustram estimativas de uma execução e devem ser lidos com o horizonte, o fuso horário e a versão do modelo.

### 4. Leitura guiada do dashboard

| Elemento | Pergunta que ajuda a responder |
| --- | --- |
| Indicadores das próximas 24 h | Qual a energia estimada no horizonte (GWh) e qual o pico de potência (MW)? |
| Curva horária de geração | Em quais horários a previsão sobe ou cai? Observe também o fuso horário informado. |
| Vento e fator de capacidade | Como a meteorologia de entrada se relaciona com a utilização prevista da capacidade instalada? |
| Identidade e explicabilidade | Qual versão gerou a previsão e quais variáveis contribuíram? A explicabilidade descreve as associações aprendidas pelo modelo. |

### 5. Acompanhamento e sinais de atenção

O monitoramento distingue mudanças na meteorologia de mudanças no desempenho. Essa separação ajuda a investigar desvios e orientar decisões de manutenção do modelo. A avaliação do erro utiliza horas com geração observada disponível e válida, mantendo a comparação rastreável.

A primeira janela, registrada em 05/10/2026, reuniu 96 horas meteorológicas e 72 com geração observada. Esse acompanhamento inaugura a avaliação contínua da operação. As revisões registram período, cobertura, erro e versão utilizada, construindo evidências de estabilidade ao longo do tempo.

## Governança e próximos marcos

### 6. O que já foi demonstrado

A reprodução em um cluster separado, com restauração de estado e preservação da v17, demonstra uma conquista operacional concreta. Um lote sintético de três horas produziu previsões idênticas às da origem, verificando a consistência do serving nas condições testadas.

O recibo v1.0 registra 225 testes de software aprovados, um ignorado e dois avisos; o teste ignorado foi aprovado no job específico de infraestrutura. Somam-se seis testes Terraform locais e sete testes AWS com mocks. Essas evidências de qualidade de software complementam as métricas preditivas e fortalecem a manutenção da solução.

A arquitetura AWS está descrita como código e testada com mocks, enquanto a execução efetiva ocorre no cluster local. Essa preparação organiza a futura implantação em nuvem, quando serão aferidos disponibilidade, latência e custos. A reprodução do modelo histórico utiliza o código público e os backups privados de dados e artefatos.

### 7. Como mudanças são controladas

O monitoramento é observacional por padrão. O retreinamento segue critérios explícitos: solicitação, mês completo, geração observada integral e sinal de mudança nos dados ou no desempenho. A promoção de um candidato passa pela comparação com o modelo atual no mesmo período temporal e pelos critérios de qualidade definidos, preservando o controle sobre mudanças.

Para adoção organizacional, propõe-se distribuir responsabilidades por dados, modelo, infraestrutura e aprovação do uso de negócio, ampliando a governança já praticada no projeto.

### 8. Próximas decisões e evidências necessárias

| Marco | Evidência necessária para a revisão |
| --- | --- |
| 7 e 14 dias completos | Cobertura de geração observada, erros por período e investigação de desvios recorrentes. |
| Outubro fechado | Reavaliação no novo mês, comparando modelos na mesma janela; setembro permanece como referência histórica. |
| Validação day-ahead | Previsões e meteorologia arquivadas na emissão, comparadas à geração observada em horas pareadas e a um baseline definido previamente. |
| Robustez operacional | Ensaios separados de recuperação, restore e rollback, com duração, resultado e identidade do modelo registrados. |

Recomendação: consolidar o acompanhamento e a validação prospectiva como próximos marcos. A revisão dos resultados orientará a ampliação do uso e da infraestrutura, com critérios explícitos de promoção.

### Base documental

- [Model Card: modelo, métricas e limites](https://github.com/wanderson42/renewable-energy-mlops/blob/8ffeed81cde9a7f690f50f9777afa00fd152aa97/docs/MODEL_CARD.md)
- [Recibo v1.0: validações e escopo](https://github.com/wanderson42/renewable-energy-mlops/blob/8ffeed81cde9a7f690f50f9777afa00fd152aa97/docs/evidence/v1_consolidation_2026-10-05.json)
- [Dashboard: captura de referência](https://github.com/wanderson42/renewable-energy-mlops/blob/8ffeed81cde9a7f690f50f9777afa00fd152aa97/notebooks/streamlit_day_ahead.png)
- [README: fontes e visão do produto](https://github.com/wanderson42/renewable-energy-mlops/blob/8ffeed81cde9a7f690f50f9777afa00fd152aa97/README.md)

Evidências do projeto: revisão 8ffeed8 da main, consultada em 06/10/2026. Resultados preditivos e operacionais são apresentados conforme os respectivos registros.

## Uso em produção e expansão

### 9. Caminho para produção

O ajuste histórico, a melhoria sobre a v10 e a operação reproduzível sustentam a proposta de um piloto de produção supervisionado. A recomendação é apoiar a análise, medir a qualidade prospectiva e ampliar o uso conforme os resultados.

| Uso pretendido | Posição recomendada |
| --- | --- |
| Operação local de portfólio | Manter serving e monitoramento, preservando a rastreabilidade. |
| Piloto de apoio à análise | Executar em paralelo ao processo atual, com revisão humana; medir desempenho prospectivo e definir limites de aceitação. |
| Despacho ou comercialização | Ampliar o uso após validação sazonal, ganho sobre baselines e tolerância a erros definida pelo usuário. |

A IEA Wind recomenda amostras representativas e métricas adequadas à aplicação [5]. A validação deverá abranger diferentes estações, erros por horizonte, viés e rampas, além de MAE, nMAE e R². Os critérios de aceitação serão definidos pelo uso pretendido e pelo desempenho frente aos baselines.

### 10. Escopo meteorológico e outros estados

O ML aproveita variáveis selecionadas de modelos meteorológicos via API [6], mantendo o escopo de dados viável para o portfólio. A literatura demonstra previsão regional com vento agregado espacialmente [7], oferecendo uma referência para evoluir a cobertura geográfica das entradas.

A extração por estado e coordenadas favorece a adaptação da plataforma. A Bahia concentra a implementação atual; novos estados poderão ser atendidos com capacidade instalada, interface, retreinamento e validação próprios. A expansão proposta é da plataforma, com desempenho verificado em cada região.

### 11. Potencial de complementaridade eólica–solar

Gomes (UFSC, 2021) observou geração solar maior perto do meio-dia e eólica à noite nos conjuntos baianos estudados [8]. Costa e Silva (2024), incluindo Morro do Chapéu Sul, identificaram complementaridade nos perfis horários médios, usando dados ONS de 2020–2022 [9].

A complementaridade abre uma frente de evolução: integrar previsão eólica e solar e avaliar estabilidade e valor operacional da geração combinada. A extensão proposta deverá medir os benefícios por local e escala temporal. Almeida e Toni (2026) acrescentam contexto territorial e socioambiental [10].

### Referências técnicas e territoriais

- [[1] scikit-learn — Definição e interpretação de R²](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
- [[2] Open-Meteo — Previous Runs: previsões em horizontes fixos](https://open-meteo.com/en/docs/previous-runs-api)
- [[3] SDE Bahia (set. 2026) — Informe executivo de energia eólica](https://www.ba.gov.br/sde/sites/site-sde/files/2026-09/Informe%20Executivo%20de%20Energia%20E%C3%B3lica_%20Setembro%202026.pdf)
- [[4] Enel (2021) — Início da operação de Morro do Chapéu Sul II](https://www.enel.com.br/pt-saopaulo/midia/news/d202111-parque-eolico-morro-chapeu-sul-inicia-operacao.html)
- [[5] Möhrlen et al. (2023) — IEA Wind: exemplos de avaliação de previsões](https://iea-wind.org/wp-content/uploads/2024/10/WIW2023_092_IEAWindRPExmples_paper.pdf)
- [[6] Open-Meteo — Weather Forecast API e fontes meteorológicas](https://open-meteo.com/en/docs)
- [[7] Thorarinsdottir et al. (2019) — Previsão eólica regional agregada](https://arxiv.org/abs/1903.01186)
- [[8] Gomes (UFSC, 2021) — Complementaridade eólica–fotovoltaica no Nordeste](https://repositorio.ufsc.br/handle/123456789/228475)
- [[9] Costa e Silva (CBENS, 2024) — Complementaridade na Bahia](https://doi.org/10.59627/cbens.2024.2490)
- [[10] Almeida e Toni (NERA, 2026) — Expansão eólica e híbrida na Bahia](https://doi.org/10.1590/1806-675520262911389)
