# v1.0 — IaC e deployment reproduzível

## Estado desta etapa

Auditoria inicial realizada em 05/10/2026 na branch
`infra/reproducible-deployment`, sobre o commit
`3490b6503d1e2284f71ff6b3bcd63fef774bc04a`.

Esta etapa registra o contrato de reprodução e os passos necessários para
implementá-lo. O KinD de ensaio foi criado e validado pelo operador em
05/10/2026. A instalação dos workloads, Terraform e o deployment automatizado
ainda não foram executados. O rollout operacional permanece manual.

A v0.3 continua em acompanhamento longitudinal. Seu cluster e seu Registry são
a referência operacional; os ensaios de reprodução usam um cluster separado.

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

- [x] Inventário da v0.3 capturado, incluindo imagens dos workloads, imagem do node e versão do KinD.
- [x] Configuração KinD reproduzida em um cluster separado; node Ready e StorageClass disponível.
- [ ] Instalação dos workloads reproduzida no cluster separado.
- [ ] Operação simultânea dos dois ambientes sem conflito de portas/processos.
- [ ] Buckets e dados mínimos disponíveis no ambiente de ensaio.
- [ ] Snapshot de metadata e artefatos restaurado e identidade do modelo verificada.
- [ ] Gate HTTP e smoke de referência aprovados no destino.
- [ ] Terraform instala a release e reaplicação não produz mudanças inesperadas.
- [ ] Atualização por digest e rollback exercitados com validação.
- [ ] Automação de deployment executada de ponta a ponta.
- [ ] Blueprint AWS validado e testado com mocks, com limitações explícitas.

## 7. Inventário coletado e perfil Helm

O inventário fornecido pelo operador em 05/10/2026 está registrado em
[`evidence/deployment_baseline_2026-10-05.json`](evidence/deployment_baseline_2026-10-05.json).
Ele confirma os quatro workloads com uma réplica disponível, os dois PVCs
`Bound`, Kubernetes `v1.37.0` e o champion v17 associado à Run
`1d13a61244c54f06aa70f43a9993ea37`.

O chart passa a aceitar `image.repository`, `image.tag`, `image.digest`
e `image.pullPolicy` nos quatro componentes. Quando fornecido, o digest
prevalece sobre a tag. Os valores padrão mantêm as referências anteriores;
o perfil de ensaio fixa os digests observados em
[`helm/environments/repro.yaml`](../helm/environments/repro.yaml).

Esses digests serão usados no ensaio na máquina de origem; sua disponibilidade
no registry ainda precisa ser verificada. Nenhuma imagem foi baixada nem
nenhum workload foi aplicado como parte desta alteração.

Validação local do chart, sem acesso a Kubernetes:

```bash
helm lint helm --strict
helm lint helm --strict -f helm/environments/repro.yaml
python3 -m unittest discover -s tests/infra -p test_helm_chart.py -v
```

O workflow `Helm chart validation` executa essas verificações em pushes nas
branches `infra/**` e na `main`, e em PRs para `main`, quando os arquivos
correspondentes mudam. Ele não aplica manifests nem publica imagens.

O runtime do MLflow ainda instala dependências sem versões no startup. Fixar
o digest do contêiner resolve a identidade da imagem, mas não essa instalação;
a correção continua pendente antes de declarar reprodução completa.

O operador também confirmou os cinco testes iniciais de renderização na
máquina de origem. O aviso de lint sobre ícone recomendado é informativo;
os dois perfis passaram em `helm lint --strict`.

### Criação do KinD separado

O node foi identificado como
`kindest/node:v1.37.0@sha256:a1ed56cfb0e7b93589bdf97c8cd566405a265939e3620fc4f5de89adff580ae5`.
A versão do CLI observada é
`kind v0.34.0-alpha+aa74c7f3e55dba go1.26.7 linux/amd64`.
O ensaio inicial utiliza esse tooling existente. A distribuição do CLI para
reprodução por terceiros ainda precisa ser definida e validada; a versão alpha
não é apresentada como release estável.

A configuração do novo cluster está em
[`infra/kind/repro.yaml`](../infra/kind/repro.yaml). Ela usa um control-plane,
node fixado por digest e API ligada ao loopback. Não configura portas de
aplicação nem mounts para diretórios do cluster operacional.

Criar somente o cluster de ensaio:

```bash
make repro-cluster
```

O target verifica Docker, KinD e kubectl, recusa recriar um cluster existente e
recusa sobrescrever um kubeconfig existente. Usa explicitamente o runtime
Docker, o nome `energy-mlops-repro` e o arquivo `.repro/kubeconfig`, protegido
com permissões locais e ignorado pelo Git. Não utiliza o `make ports`.

Após criar o cluster, conferir seu acesso e a referência operacional:

```bash
kubectl --kubeconfig .repro/kubeconfig \
  --context kind-energy-mlops-repro get nodes -o wide

kubectl --kubeconfig .repro/kubeconfig \
  --context kind-energy-mlops-repro get storageclass

kubectl config current-context
curl -fsS --max-time 5 http://localhost:8000/model-info
```

O último contexto deve continuar sendo o da origem (`kind-energy-mlops`, no
ambiente inventariado). O uso de `--kubeconfig` explícito pelo bootstrap evita
alterar o kubeconfig operacional. A resposta da API de origem deve continuar
identificando v17 e a Run documentada.

O bootstrap ainda não instala workloads nem restaura dados. Se a criação falhar,
inspecionar o cluster e `.repro/kubeconfig` antes de tentar novamente; o script
não faz exclusão nem limpeza automática de recursos.

### Resultado do bootstrap real

A criação executada pelo operador em 05/10/2026 confirmou:

- os clusters `energy-mlops` e `energy-mlops-repro` disponíveis;
- node `energy-mlops-repro-control-plane` Ready, Kubernetes `v1.37.0`;
- StorageClass `standard` disponível no destino;
- contexto operacional preservado como `kind-energy-mlops`;
- API de origem ainda servindo v17 e a mesma Run.

A evidência está em
[`evidence/reproduction_cluster_2026-10-05.json`](evidence/reproduction_cluster_2026-10-05.json).
Ela comprova a criação isolada do cluster, não a reprodução do sistema MLOps
completo. Não é necessário executar `make repro-cluster` novamente.

### Próxima etapa: infraestrutura antes do serving

O chart `0.1.2` inclui `api.enabled`, com padrão `true`. O perfil
[`helm/environments/repro-bootstrap.yaml`](../helm/environments/repro-bootstrap.yaml)
desabilita a API durante a preparação de PostgreSQL, RustFS e MLflow. Ele
complementa o perfil `repro.yaml`; não contém imagens ou credenciais próprias.

O serving será habilitado na mesma release depois que os dados, o Registry e
os artefatos estiverem disponíveis. Esse preparo não muda o alias da origem.

Antes de aplicar a infraestrutura, capturar as versões de Python/MLflow e das
dependências S3/PostgreSQL instaladas no contêiner de origem. O `poetry.lock`
do aplicativo não comprova as versões instaladas pelo startup do servidor
MLflow. Essa coleta orientará a eliminação das instalações sem versões.

Também verificar `terraform version` e `helm version --short` na máquina de
origem. O próximo provisionamento será preparado pelo root module Terraform
local, evitando instalar uma release manualmente e precisar importá-la depois.

O payload exato do smoke histórico também permanece pendente. O notebook de
governança registra seu timestamp (`2026-10-04T12:00:00Z`) e a saída, mas não
contém todos os inputs necessários para reconstruir essa requisição.

## Referências do projeto

- [Operação atual](OPERATIONS.md)
- [Infraestrutura e limites de persistência validados](INFRASTRUCTURE.md)
- [Contrato e governança do modelo](MODEL_CARD.md)
