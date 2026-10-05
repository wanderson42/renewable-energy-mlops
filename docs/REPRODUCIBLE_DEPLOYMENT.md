# v1.0 — IaC e deployment reproduzível

## Estado desta etapa

Auditoria inicial realizada em 05/10/2026 na branch
`infra/reproducible-deployment`, sobre o commit
`3490b6503d1e2284f71ff6b3bcd63fef774bc04a`.

Esta etapa registra o contrato de reprodução e os passos necessários para
implementá-lo. O provisionamento em um KinD limpo, Terraform e o deployment
automatizado ainda não foram executados. O rollout operacional permanece manual.

A v0.3 continua em acompanhamento longitudinal. Seu cluster e seu Registry são
a referência operacional; os ensaios de reprodução serão feitos em um cluster
separado.

## 1. Escopo acordado

| Trilha | Entrega | Limite da evidência |
|---|---|---|
| Local | Sistema executável em KinD, instalação declarativa por Terraform/Helm, validação e rollback | Deployment local realmente exercitado |
| AWS | Arquitetura Terraform para VPC, EKS, S3, RDS e IAM | Validação estática e testes com mocks; sem alegação de deployment AWS |

O projeto não depende de despesas com recursos AWS. LocalStack é opcional e não
é requisito para concluir a v1.0. O deployment local e o blueprint AWS serão
entregues nessa ordem, reaproveitando o chart e os gates existentes.

## 2. Proveniência da base

O commit `3490b65` contém a otimização Docker. A revisão da imagem validada,
`ad53ea76f0a735579a5c30bbef73a90cf416d51f`, é o merge do PR #3 dessa otimização
na `main`. A comparação entre os dois commits não apresenta arquivos diferentes.

Isso explica a diferença entre os identificadores de commit. Não identifica,
por si só, o digest da imagem em execução: esse valor deve ser coletado do
ambiente operacional e registrado separadamente.

## 3. Resultado da auditoria do repositório

| Área | Estado encontrado | Trabalho necessário |
|---|---|---|
| Bootstrap do cluster | README e runbook começam com os workloads já provisionados; não há configuração KinD versionada | Versionar criação do cluster de ensaio e configuração do kubeconfig |
| API | Chart usa `repository:tag`, com `latest`; `imagePullPolicy` está fixo no template | Permitir digest e respeitar a configuração de pull policy |
| Demais imagens | RustFS usa `latest`; PostgreSQL usa `13`; MLflow usa `v3.16.1-full` | Inventariar as imagens executadas e fixar referências reproduzíveis para o ensaio |
| Runtime MLflow | Instala `psycopg2-binary` e `boto3` sem versões a cada início do contêiner | Definir uma instalação com versões fixas, preferencialmente na construção de uma imagem |
| Isolamento Kubernetes | Makefile e supervisor usam o contexto e namespace implícitos | Configurar contexto e namespace explicitamente nas operações do ensaio |
| Processos locais | `make ports` chama `stop-ports`, que usa `pkill` global para forwards, Prefect e Streamlit | Isolar processos e portas antes de operar dois ambientes simultaneamente |
| Prefect e dashboard | São processos locais iniciados por Poetry, não workloads do chart | Incluir sua configuração e inicialização no contrato de reprodução local |
| Storage | PVCs de PostgreSQL e RustFS existem; recuperação após destruição do cluster não foi demonstrada | Exercitar backup/restauração em ambiente separado |
| Dados | Credenciais, datasets reais e estado do Registry não fazem parte do Git | Documentar preparação de buckets, dados mínimos e bootstrap/restauração do modelo |
| Saúde da API | Chart não configura readiness probe; `/health` depende do champion carregado | Separar readiness de Kubernetes e validação do modelo pelo gate HTTP |
| CI/CD | Actions testa e publica a imagem em pushes na `main`; não executa deployment | Encadear digest, aplicação, rollout, gate e evidências depois de validar o deployment local |
| Documentação | `docs/OPERATIONS.md` ainda apresenta resultado histórico de 75 testes e um fluxo resumido de CT | Atualizar o runbook durante a implementação, respeitando os guards atuais do código |

Um rollout Kubernetes bem-sucedido não comprova que o champion está carregado.
O ensaio somente passa quando saúde, identidade do modelo e inferência forem
validados.

## 4. Contrato de reprodução local

### Infraestrutura

- Cluster de ensaio com nome próprio, sugerido: `energy-mlops-repro`.
- Kubeconfig dedicado, contexto explícito e namespace explícito.
- Release Helm com configuração versionada e credenciais locais separadas.
- Imagens identificadas por digest no ensaio de aceitação.
- Portas locais diferentes das utilizadas pela v0.3.
- Cada objeto gerenciado por uma ferramenta: Terraform controla a release Helm;
  Helm controla os objetos definidos pelo chart.

A criação do KinD pode continuar sendo um procedimento simples versionado.
Não será necessário introduzir um provider de terceiros somente para criar o
cluster. O Terraform administrará a instalação depois que o cluster existir.

### Dados e modelo

Uma instalação vazia de MLflow não contém o alias `@champion`. Existem dois
ensaios diferentes:

| Ensaio | Objetivo | Identidade esperada |
|---|---|---|
| Bootstrap novo | Criar os buckets, preparar os dados necessários e executar o lifecycle até obter um modelo servível | Nova Run e versão registradas; não exigir os identificadores históricos |
| Restauração da v0.3 | Recuperar um snapshot consistente de PostgreSQL e dos objetos necessários em RustFS | Preservar a v17, a Run histórica e o alias, verificando também os artefatos |

O primeiro ensaio de equivalência com o smoke histórico usará a restauração da
v0.3. O backup deve ser preparado sem alterar o alias ou os dados da origem.
Restaurar apenas o PostgreSQL é insuficiente: os artefatos referenciados também
precisam existir e estar acessíveis no destino.

O payload exato do smoke e o critério numérico de comparação precisam ser
preservados. O valor histórico informado é
`predicted_mw = 2092.931369766858`; um valor isolado não permite reproduzir a
inferência sem os respectivos inputs e a identidade do modelo.

### Operação

O fluxo final desejado é:

1. Receber uma referência de imagem por digest.
2. Aplicar a configuração declarativa da release.
3. Aguardar o rollout.
4. Verificar `/health`, `/model-info` e o smoke de inferência.
5. Registrar digest solicitado, imagens executadas, modelo e resultados.
6. Permitir retorno à configuração anterior e repetir o gate.

Terraform executado manualmente é uma entrega intermediária. A atualização
automatizada só será considerada concluída depois que o fluxo acima estiver
encadeado e exercitado com acesso explícito ao cluster.

## 5. Próxima coleta no ambiente operacional

Os comandos abaixo são de leitura. Ajuste o contexto e o namespace para os
valores reais do ambiente v0.3. Não execute `make ports`, `make stop-ports` ou
comandos de instalação para realizar esta coleta.

Primeiro identifique o contexto:

```bash
kubectl config current-context
kubectl config get-contexts -o name
```

Depois defina os valores e consulte o ambiente:

```bash
V0_3_CONTEXT="SUBSTITUA_PELO_CONTEXTO_REAL"
V0_3_NAMESPACE="default"

helm list --kube-context "$V0_3_CONTEXT" --namespace "$V0_3_NAMESPACE"

kubectl --context "$V0_3_CONTEXT" --namespace "$V0_3_NAMESPACE" \
  get deployments \
  -o 'custom-columns=NAME:.metadata.name,IMAGES:.spec.template.spec.containers[*].image,READY:.status.readyReplicas'

kubectl --context "$V0_3_CONTEXT" --namespace "$V0_3_NAMESPACE" \
  get pods \
  -o 'custom-columns=NAME:.metadata.name,IMAGES:.spec.containers[*].image,IMAGE_IDS:.status.containerStatuses[*].imageID,READY:.status.containerStatuses[*].ready'

kubectl --context "$V0_3_CONTEXT" --namespace "$V0_3_NAMESPACE" \
  get pvc \
  -o 'custom-columns=NAME:.metadata.name,STATUS:.status.phase,CLASS:.spec.storageClassName,SIZE:.spec.resources.requests.storage'
```

O `imageID` pode identificar uma imagem específica da plataforma, enquanto a
referência solicitada pode apontar para um manifesto de múltiplas plataformas.
Ambos devem ser registrados sem presumir igualdade textual entre eles.

Também precisamos preservar localmente o payload do smoke histórico e
identificar o procedimento de backup disponível. Secrets, kubeconfigs,
dumps do banco e artefatos do modelo não devem ser adicionados ao Git.

## 6. Critérios de aceitação

- [ ] Inventário da v0.3 capturado, incluindo imagens realmente executadas.
- [ ] Configuração KinD e instalação Helm reproduzidas em um cluster separado.
- [ ] Operação simultânea dos dois ambientes sem conflito de portas/processos.
- [ ] Buckets e dados mínimos disponíveis no ambiente de ensaio.
- [ ] Snapshot de metadata e artefatos restaurado e identidade do modelo verificada.
- [ ] Gate HTTP e smoke de referência aprovados no destino.
- [ ] Terraform instala a release e reaplicação não produz mudanças inesperadas.
- [ ] Atualização por digest e rollback exercitados com validação.
- [ ] Automação de deployment executada de ponta a ponta.
- [ ] Blueprint AWS validado e testado com mocks, com limitações explícitas.

## Referências do projeto

- [Operação atual](OPERATIONS.md)
- [Infraestrutura e limites de persistência validados](INFRASTRUCTURE.md)
- [Contrato e governança do modelo](MODEL_CARD.md)

