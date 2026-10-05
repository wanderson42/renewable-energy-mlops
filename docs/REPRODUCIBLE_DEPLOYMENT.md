# v1.0 — IaC e deployment reproduzível

## Estado desta etapa

Auditoria inicial realizada em 05/10/2026 na branch
`infra/reproducible-deployment`, sobre o commit
`3490b6503d1e2284f71ff6b3bcd63fef774bc04a`.

Esta etapa registra o contrato de reprodução e os passos necessários para
implementá-lo. O KinD de ensaio foi criado e validado pelo operador em
05/10/2026. O primeiro `terraform apply` instalou PostgreSQL, RustFS e MLflow no ensaio,
com a API desabilitada. O snapshot foi restaurado, a identidade da v17 foi
confirmada no Registry do destino e o plano posterior retornou `No changes`.
O procedimento de habilitação e gate HTTP foi executado: a API do ensaio serve
a v17 e reproduziu exatamente o lote pareado da origem. O plano posterior
retornou `No changes` com serving habilitado. O rollout operacional da origem
permanece manual. O ensaio de falha de pull e recuperação declarativa também
passou: digest validado, v17/Run, inferência e UIDs dos PVCs preservados, com
plano final `No changes`. A imagem própria do runtime MLflow, a configuração
dos processos locais e o blueprint AWS continuam pendentes.

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
- [x] PostgreSQL, RustFS e MLflow provisionados por Terraform no cluster separado, com PVCs Bound.
- [x] Serving habilitado e validado após a restauração no cluster separado.
- [x] Operação simultânea das APIs de origem e ensaio sem conflito de portas/processos.
- [x] Buckets e dados mínimos disponíveis no ambiente de ensaio.
- [x] Snapshot de metadata e conteúdo dos dois buckets capturado e verificado.
- [x] Snapshot restaurado e identidade do modelo verificada no destino.
- [x] Gate HTTP e lote pareado `paired_synthetic_batch_v1` aprovados no destino.
- [x] Terraform instala a release e um novo plano após o apply não apresenta mudanças.
- [x] Falha de pull por digest e recuperação declarativa exercitadas com gate e PVCs preservados.
- [ ] Upgrade para uma nova imagem funcional validado; o ensaio de pull não cobre esse caso.
- [ ] Runtime MLflow empacotado e validado em imagem própria por digest.
- [ ] Configuração e inicialização de Prefect/dashboard incluídas na reprodução local.
- [x] Automação de deployment local executada: plano, apply, rollout e gate.
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

O chart `0.1.3` fixa as nove versões coletadas no runtime do servidor MLflow.
Ainda há instalação via pip no startup, com dependência de rede/PyPI e sem
hashes de wheels ou congelamento de todas as dependências transitivas.
Empacotar esse runtime em imagem própria e validá-la por digest permanece
pendente antes de declarar reprodução completa.

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

O chart, desde `0.1.2`, inclui `api.enabled`, com padrão `true`. O perfil
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

## 8. Primeiro root module Terraform local

O inventário do servidor confirmou Python `3.11.15` e as nove versões agora
registradas no baseline. Esse ambiente pertence ao contêiner MLflow; o ambiente
Python `3.14` do aplicativo continua gerenciado pelo Poetry e pelo seu lockfile.
Não é necessário adicionar Terraform ao `pyproject.toml` ou trocar o Python do
projeto. Os testes Python também podem ser executados com `poetry run python`.

O root module está em `terraform/environments/local/`. Ele administra uma única
`helm_release` no cluster já criado:

| Configuração | Valor |
|---|---|
| Terraform CLI | `1.16.5` |
| Provider Helm | `3.3.0`, com lockfile versionado |
| Kubeconfig | `.repro/kubeconfig` |
| Contexto | `kind-energy-mlops-repro` |
| Namespace e release | `energy-mlops-repro` |
| Imagens | Digests do perfil `repro.yaml` |
| API | Desabilitada, inclusive se o arquivo de credenciais definir `api.enabled: true` |
| State local | `.repro/terraform.tfstate` |

O provider utiliza a biblioteca Helm, sem invocar o CLI Helm instalado no host.
O Helm `v4.3.0` informado pelo operador pode continuar sendo usado para inspeção;
a CI de renderização utiliza `3.19.0`. A instalação real via provider foi
validada no ensaio, conforme a evidência de bootstrap abaixo.

As credenciais são lidas do arquivo local já utilizado pelo projeto,
`helm/values_secrets.yaml`. Ele deve conter `postgres.user`, `postgres.password`,
`postgres.db`, `rustfs.rootUser` e `rustfs.rootPassword`, todos preenchidos.
O arquivo não deve ser enviado nem incluído no Git. Marcar o input como sensitive
não impede que credenciais reapareçam em `metadata.values` do provider Helm:
isso foi observado no plano real. State, plano binário, metadata e logs brutos
podem armazenar as credenciais. São arquivos privados do ensaio, não evidências
publicáveis. O bootstrap protege `.repro/` e os scripts utilizam `umask 077`;
não habilitar logs de debug nem publicar esses arquivos.

### Instalar somente o Terraform do ensaio

Na raiz do repositório:

```bash
make repro-tools
.repro/bin/terraform version
```

Esse target baixa o binário oficial Linux AMD64 `1.16.5`, compara o SHA256 do
arquivo ZIP com o checksum oficial fixado no script e instala em `.repro/bin/`.
Usa apenas ferramentas do sistema e a biblioteca padrão de `python3`; não
instala pacotes Python, não exige sudo e não altera o ambiente Poetry.

### Gerar o primeiro plano

```bash
make repro-plan
```

O target confere o cluster de ensaio, executa `fmt -check`, `init` com o lockfile,
`validate` e `plan`. Não executa `apply`. A primeira execução deve propor a
criação de **uma `helm_release.mlops`**, sem alterações ou destruições. Os
workloads internos do chart não aparecem como recursos Terraform separados.
A saída deve identificar `energy-mlops-repro`, e `api.enabled` deve ser `false`.
Se aparecer uma release existente ou uma substituição, investigar a origem do
state e os recursos no destino antes de continuar.

O plano binário fica em `.repro/bootstrap.tfplan`. Ele é vinculado a este checkout
local: alterações no chart, nos perfis ou nas credenciais exigem novo plano antes
da aplicação. Não transportar esse plano para outra máquina.

Após validar esse resultado, o provisionamento usará o plano salvo:

```bash
.repro/bin/terraform -chdir=terraform/environments/local apply ../../../.repro/bootstrap.tfplan

kubectl --kubeconfig .repro/kubeconfig --context kind-energy-mlops-repro \
  --namespace energy-mlops-repro get deployments,pods,pvc
```

O timeout de espera é de dez minutos. Não há exclusão automática em caso de
falha; inspecionar os pods e a release antes de uma nova tentativa. Não usar
`helm upgrade`, `helm rollback` ou `terraform destroy` por fora do fluxo durante
a restauração: Terraform é o responsável pela release, e os PVCs contêm os dados
do ensaio.

Os readiness checks aguardam PostgreSQL aceitar conexões e MLflow/RustFS
escutarem suas portas. Isso ainda não valida buckets, autenticação S3, artefatos,
Registry nem inferência. Esses gates serão executados durante a restauração.

### Validação em CI, sem ambiente real

```bash
.repro/bin/terraform -chdir=terraform/environments/local init -backend=false -lockfile=readonly
.repro/bin/terraform -chdir=terraform/environments/local validate
.repro/bin/terraform -chdir=terraform/environments/local test
```

O workflow `Terraform local validation` usa um provider Helm simulado e
credenciais fictícias. Seus testes verificam o destino dedicado, a precedência
do bloqueio da API e a rejeição de credenciais ausentes. Não acessa Kubernetes
nem demonstra provisionamento real. O `plan` operacional é uma etapa distinta,
executada na máquina que possui o kubeconfig do ensaio.

## 9. Resultado do primeiro apply e preparo da restauração

O operador executou o plano salvo em 05/10/2026: `1 added, 0 changed, 0 destroyed`,
com criação da release em 52 segundos. PostgreSQL, RustFS e MLflow apresentaram
`1/1 Ready`, zero reinícios e os PVCs de 2 GiB/5 GiB estavam `Bound`.
A evidência está em
[`evidence/terraform_bootstrap_2026-10-05.json`](evidence/terraform_bootstrap_2026-10-05.json).

Esse marco comprova o provisionamento real da infraestrutura pelo Terraform.
Um segundo plano apresentou `No changes`, e os dois testes Terraform com mocks
passaram na máquina do operador. A restauração posterior do Registry está
documentada na seção 12; a equivalência de inferência no lote pareado foi
demonstrada posteriormente, conforme a seção 14.
Até a conclusão dessa restauração, a API permaneceu desabilitada.
O serving foi exercitado posteriormente. A reprodução do sistema completo
ainda exige as demais fronteiras de aceitação descritas neste documento.

Antes de preparar o snapshot:

```bash
make repro-plan
make repro-inventory
kubectl config current-context
curl -fsS --max-time 5 http://localhost:8000/model-info
```

O segundo plano deve informar `No changes`. Não aplicar esse plano vazio.
As alterações desta etapa não modificam o chart ou o root module Terraform.
O último par de comandos confere o contexto e a identidade servida pela origem.

O inventário executa o script Python via stdin dentro dos contêineres MLflow de
origem e destino. Usa o Python e os pacotes do servidor; não depende do ambiente
Poetry nem instala dependências. O script:

- consulta o Registry, o alias champion, a Run e os URIs do modelo;
- exige a identidade v17/Run da origem documentada;
- confere que o Registry do destino está vazio;
- lista metadata dos objetos para contar arquivos e bytes dos dois buckets;
- recusa continuar se houver dados de aplicação no destino;
- mostra as versões dos pacotes e os image IDs dos pods do ensaio.

As chamadas são de leitura. Não cria buckets, não baixa objetos, não faz dump de
PostgreSQL e não altera aliases. Os resultados ficam em
`.repro/source-data-inventory.json` e `.repro/destination-data-inventory.json`.
Uma falha em qualquer lado interrompe o preflight; os snapshots e a restauração
serão preparados depois de revisar essas evidências.

A restauração usará um dump lógico do banco e cópias dos objetos, preservando os
identificadores e os caminhos referenciados pelo MLflow. Os PVCs da origem não
serão conectados ao ensaio. Uma captura consistente também exige definir um
intervalo sem escritores concorrentes; essa condição será estabelecida antes
do backup, sem assumir que DB e object storage têm snapshot transacional comum.

## 10. Preflight confirmado e captura do snapshot

O inventário real confirmou que o destino tem Registry vazio e ainda não possui
os buckets de aplicação. O runtime MLflow é igual ao da origem nas nove versões
coletadas, e os três image IDs do ensaio correspondem aos digests do baseline.
O contexto operacional e a identidade servida pela API de origem foram
preservados. A evidência está em
[`evidence/data_preflight_2026-10-05.json`](evidence/data_preflight_2026-10-05.json).

| Bucket na origem | Objetos | Bytes |
|---|---:|---:|
| `energy-lake` | 19 | 20.949.616 |
| `mlflow-artifacts` | 232 | 1.808.792.024 |
| Total | 251 | 1.829.741.640 |

A v17 aponta para o Logged Model `m-b5741525f9ce41c98d166288955e9f6b`.
Seus artefatos estão em
`s3://mlflow-artifacts/6/models/m-b5741525f9ce41c98d166288955e9f6b/artifacts`,
enquanto os artefatos da Run têm outro prefixo. A captura preserva o banco inteiro
e o conteúdo atual dos dois buckets, incluindo esse Logged Model e as demais
versões presentes, sem registrar novamente o champion.

### Condição para a captura

Execute em um intervalo sem escritores concorrentes: nenhum treino, promoção,
ingestão, monitoring que exporte dados, limpeza ou alteração manual deve gravar
no MLflow/RustFS durante a captura. A API pode continuar atendendo inferências.
O script não pausa tarefas Prefect nem altera serviços da origem.

`pg_dump` produz um snapshot consistente do banco individualmente. PostgreSQL e
S3 não têm uma transação comum. O intervalo sem escritores é a condição
operacional para capturá-los juntos; as comparações de listing e identidade
antes/depois ajudam a detectar alterações, mas não comprovam a ausência de toda
mutação transitória ou de mudanças somente no banco.

### Capturar e verificar

Na raiz do projeto:

```bash
make repro-backup
```

O target usa contexto explícito para todas as chamadas Kubernetes. Ele refaz o
preflight, confirma o destino vazio e grava um novo diretório privado em
`.repro/snapshots/`. Em seguida:

1. Lista keys, tamanhos, ETags e datas dos objetos da origem, confirmando o
   `MLmodel` no prefixo da v17.
2. Usa o `pg_dump` do próprio PostgreSQL de origem para criar `postgres.dump`
   em formato custom, com schema e dados.
3. Lê os objetos S3 em fluxo e grava `objects.tar`, com metadata e SHA256 por
   objeto. Não precisa guardar o conjunto inteiro em RAM ou no filesystem do
   contêiner MLflow.
4. Compara novamente o listing e a identidade do modelo. Se houver mudança,
   interrompe o backup antes de finalizar seu manifesto.
5. Usa `pg_restore --file=/dev/null` para ler/descomprimir o dump e gerar SQL
   descartado, sem conectar ao banco nem executar SQL.
6. Confere a identidade da API de origem e valida os hashes dos objetos no host.
7. Grava `summary.json`, `SHA256SUMS` e o caminho em `.repro/latest-snapshot`.

A validação no host utiliza a biblioteca padrão de `python3`. As chamadas S3
utilizam boto3 dentro do servidor MLflow, com as credenciais já configuradas no
contêiner. O ambiente Poetry da aplicação permanece como estava.

O script exige espaço livre para o tamanho listado dos objetos mais uma reserva
de 2 GiB para o banco. Esse limite não substitui a verificação do espaço quando
os dados crescerem. O arquivo TAR é sem compressão; o dump custom do banco usa
a compressão padrão do PostgreSQL.

Não há gravação no banco/object storage de origem ou destino nessa etapa.
Os backups contêm dados e metadata privados: `.repro/` permanece ignorado pelo
Git. Envie somente a saída textual e o resumo, preservando os arquivos na máquina.

Se a captura falhar, o diretório mantém o marcador `INCOMPLETE` e os arquivos
parciais, sem atualizar o caminho do último snapshot concluído. Inspecione a
causa antes de repetir. O script não remove backups anteriores nem limpa dados.

Os testes de snapshot exercitam exportação e verificação com clientes simulados,
rejeição de corrupção, alterações durante a cópia, arquivos incompletos e
isolamento dos comandos. A captura real concluída está registrada na seção
seguinte. A verificação do backup não substitui um teste de restauração e
inferência no destino.

## 11. Snapshot capturado e restauração no ensaio

O operador concluiu o backup `snapshot-20261005T201656Z.LOW70R`, com verificação
em `2026-10-05T20:17:08.561351+00:00`. Os 251 objetos preservam os mesmos totais
do preflight. O dump tem 167.563 bytes, e o TAR dos objetos tem 1.830.041.600
bytes, incluindo seu manifesto e padding. A evidência pública contém somente
metadata e hashes em
[`evidence/snapshot_capture_2026-10-05.json`](evidence/snapshot_capture_2026-10-05.json).
Os arquivos permanecem privados na máquina do operador.

A captura passou nos hashes, no parsing do dump e nas comparações de listing e
identidade da origem. A ausência de escritores concorrentes continua sendo
uma precondição operacional, não uma propriedade demonstrada pelos hashes.

### Preparar o restore

O target `repro-restore` usa `.repro/latest-snapshot`, confere que o caminho
pertence ao diretório deste ensaio, rejeita `INCOMPLETE` e compara os arquivos
com os hashes já registrados. Não recalcula um novo resumo para aceitar um
backup alterado. A validação também relê os hashes dos objetos dentro do TAR.

Antes de escrever no destino, ele exige:

- node `energy-mlops-repro-control-plane` no kubeconfig dedicado;
- API ausente no namespace `energy-mlops-repro`;
- Registry e buckets sem dados de aplicação;
- zero Runs, Registered Models, Model Versions, Logged Models e experimentos
  adicionais ao experimento padrão no banco do ensaio.

O SQL de preflight verifica o backend MLflow `3.16.1`, com os nomes de tabelas
correspondentes a essa versão. Este é um procedimento para a primeira
restauração em um destino vazio, não um mecanismo de sincronização contínua.

Executar na raiz do projeto:

```bash
poetry run python -m unittest discover -s tests/infra -p test_repro_restore.py -v
make repro-restore
```

### O que será alterado

Somente os dados e a suspensão temporária do MLflow no ensaio:

1. Criar os dois buckets se ainda não existirem.
2. Restaurar as mesmas keys e bytes, preservando os headers de conteúdo e
   metadata capturados. Cada objeto é validado antes do upload; o PUT usa
   `IfNoneMatch=*` para recusar sobrescrita de uma key existente.
3. Relê-los no destino para confirmar todos os hashes.
4. Suspender o MLflow do ensaio em zero réplicas e aguardar seus pods encerrarem.
5. Restaurar o dump no PostgreSQL do ensaio, com `--clean --if-exists` para
   substituir o schema vazio criado pelo bootstrap e `--single-transaction`
   para aplicar o restore do banco integralmente ou fazer rollback em erro.
6. Retornar o MLflow do ensaio a uma réplica e aguardar o rollout.
7. Verificar a identidade do champion, os URIs e novamente os hashes dos objetos
   usando o backend restaurado; conferir a API de origem.

As chamadas que alteram recursos Kubernetes ou restauram SQL usam exclusivamente
`.repro/kubeconfig`, contexto `kind-energy-mlops-repro` e namespace
`energy-mlops-repro`. A criação e cópia de objetos usam o endpoint interno do
RustFS desse mesmo namespace. O Terraform permanece responsável pela release;
a suspensão temporária do processo é uma operação de restauração de dados e
retorna ao estado declarado de uma réplica quando concluída.

O upload usa um arquivo temporário por objeto no contêiner MLflow, com checagem
de espaço para o maior objeto e reserva de 64 MiB. O TAR não é extraído para
caminhos baseados nas keys. O conjunto inteiro não precisa caber em RAM ou no
filesystem do contêiner.

### Resultado esperado e falhas

O resumo fica em `.repro/restored-data.json`. Esperamos v17, a Run histórica,
o mesmo Logged Model/URI e 251 objetos com 1.829.741.640 bytes verificados.
A API do destino continua desabilitada: carregar o modelo e exercitar a
inferência serão os gates seguintes. Não há promoção, treino ou novo registro.

Os scripts não removem dados em caso de erro. Uma falha antes de restaurar o
banco pode deixar objetos parcialmente copiados. Uma falha na fase do banco
pode deixar o MLflow do ensaio suspenso; o restore SQL é transacional. Inspecione
a fase e os dados antes de repetir. Um destino já preenchido será recusado,
inclusive depois de uma restauração bem-sucedida. Não executar limpeza manual
ou novo restore sobre esse destino sem definir sua recuperação.

Os sete testes novos verificam hashes no destino, prevenção de sobrescrita,
rejeição de snapshot alterado, rejeição de node incorreto ou banco não vazio,
escopo das operações de escala/SQL e comportamento em falha. São testes com
clientes simulados; o resultado real posterior está registrado a seguir.

## 12. Restauração real e próximo gate de serving

O operador concluiu `make repro-restore`: 251 objetos e 1.829.741.640 bytes
foram verificados no destino, o PostgreSQL foi restaurado e o Registry retornou
a v17, a Run e o URI históricos. Os três deployments voltaram a `1/1 Ready`
e os PVCs permaneceram `Bound`. O plano executado após a restauração retornou
`No changes`, confirmando o retorno ao estado declarado da infraestrutura.
A evidência está em
[`evidence/snapshot_restore_2026-10-05.json`](evidence/snapshot_restore_2026-10-05.json).
Isso não demonstra inferência nem equivalência de todas as linhas do banco.

### Habilitação explícita da API

O root Terraform aceita `api_enabled`, com padrão `false`. O override final
continua impedindo que o arquivo de credenciais determine esse estágio.
Para habilitar o serving, o procedimento exige o comprovante local de restore
e consulta a identidade atual no Registry do ensaio. Ele também confirma o
node dedicado e rejeita instalação, substituição ou destruição da release.

```bash
poetry run python -m unittest discover -s tests/infra -p test_repro_serving.py -v
.repro/bin/terraform -chdir=terraform/environments/local test
make repro-serving-plan
```

O plano fica em `.repro/serving.tfplan` e deve mostrar atualização da mesma
`helm_release.mlops`, sem criação/substituição/destruição de recursos Terraform.
O chart passa a `0.1.4`; o novo deployment usa o digest já inventariado.
Startup e readiness consultam `/health`, exigindo modelo carregado.
PostgreSQL, RustFS e seus PVCs permanecem definidos pela mesma release.

Depois de revisar o plano:

```bash
make repro-serving-apply
make repro-plan
```

O apply usa somente o plano de serving salvo e roda o gate HTTP em seguida.
Antes de aplicar, registra `api_enabled=true` em `.repro/deployment.tfvars.json`.
Esse arquivo privado é reutilizado por `make repro-plan`, preservando a intenção
de serving nos planos posteriores, inclusive se o apply/gate falhar. Não remover
o arquivo para tentar corrigir uma falha: isso mudaria a configuração desejada.
O plano posterior deve retornar `No changes`. Um plano pode ficar obsoleto se
o state mudar; nesse caso, gere um novo plano. Não aplicar o plano de bootstrap
anterior para habilitar serving.

### Comparação pareada

O gate confirma o digest do deployment e do pod Ready, `/health`, `/model-info`,
alias/version/Run e igualdade das métricas servidas. Envia o mesmo lote sintético
de três horas às APIs de origem e destino; confere datas, quantidade, números
finitos, FC em `[0, 1]`, MW não negativos e equivalência numérica. As tolerâncias
absolutas são `1e-12` para FC e `1e-9` MW, sem tolerância relativa.
Consulta novamente os metadados para rejeitar mudança durante a comparação.

Essa referência chama-se `paired_synthetic_batch_v1`, com payload registrado no
script e no resultado. Ela exercita a engenharia de features, inclusive rolling,
e o serving completo. Não é um benchmark de qualidade preditiva nem reproduz
o smoke histórico `2092.931369766858`: o payload daquele smoke não foi fornecido.

O ensaio usa port-forward em uma porta temporária de `127.0.0.1`; a origem
continua em `localhost:8000`. O script encerra somente o processo que iniciou,
inclusive em falha. O resumo fica em `.repro/serving-validation.json`, com
timestamp, payload e respostas. Para repetir somente o gate: `make repro-validate`.
Um arquivo de resultado anterior comprova apenas seu timestamp; o exit code da
execução atual determina se a nova validação passou.

A suíte de serving verifica isolamento, preservação do estágio entre planos,
recusa de planos destrutivos, falha do apply, cleanup do port-forward e gates de
identidade/inferência, redação do resumo e avaliação de inputs pelo Terraform.
O teste de avaliação usa um plano real sem providers e uma fixture da atualização
Helm, sem cluster. Ele roda quando o CLI Terraform está disponível no PATH ou em
`.repro/bin/terraform`; caso contrário, é marcado como skip.
O provider Helm não inicia no runtime do editor, mas o operador já aprovou
`validate` e os três testes Terraform com mocks na revisão `c6d8d09`.
O serving real passou na revisão `1adebdb`, conforme a seção 14. O JSON do plano é inspecionado em memória e
não é impresso integralmente, pois contém valores sensíveis. Não publicar state,
plano binário ou arquivos privados de `.repro/`.
O gate não executa treino, promoção ou reload. O apply comum preserva falhas
para diagnóstico; o novo roteiro de recuperação controlada descrito na seção 14
tenta restaurar o checkpoint após sua falha intencional. Seu exercício real
passou na revisão `ca7830b`, conforme a seção 15.

## 13. Correção do review do plano e proteção da saída

Na execução real de `c6d8d09`, os onze testes Python e os três testes Terraform
passaram. O Terraform gerou um plano válido: atualização da mesma release,
API `false -> true`, chart `0.1.3 -> 0.1.4` e zero criação/destruição de recursos
Terraform. O wrapper recusou esse plano antes de aplicar.
O erro estava na checagem Python: esperava booleano no input bruto do JSON.
O Terraform `1.16.5` conserva `-var=api_enabled=true` como string nesse campo;
o output avaliado é booleano. A correção verifica esse output, o destino e o
setting final de Helm, preservando a recusa de instalação/substituição/destruição.
A causa foi reproduzida com o CLI real, sem provider nem cluster.
A evidência pública sanitizada está em
[`evidence/serving_plan_review_2026-10-05.json`](evidence/serving_plan_review_2026-10-05.json).

O plano também mostrou credenciais dentro de `metadata.values`, apesar da
marcação sensitive no input. O procedimento passa a guardar o stdout/stderr
bruto de plan e apply em `.repro/terraform-plan.log` e
`.repro/terraform-apply.log`, com permissões privadas. A revisão pública usa
somente campos permitidos: ações, contagens, confirmação de release/namespace,
versão do chart, estágio da API, hash do chart e indicação de mudança nos inputs.
Não imprime metadata nem conteúdo dos inputs, inclusive em falha do plan.
Essa mudança não remove credenciais de snapshots/state/logs já existentes.
Não publicar saídas brutas de `terraform plan`, `terraform show -json` ou logs
do provider. Em caso de erro, inspecione os arquivos privados localmente e
compartilhe somente o diagnóstico sanitizado.

Após atualizar e rodar os quinze testes de serving, gere um novo
`make repro-serving-plan`. O resumo deve continuar em `0/1/0` e confirmar o
destino de ensaio e `api.enabled=true`. Depois, `make repro-serving-apply`
executa o plano e o gate; `make repro-plan` deve retornar `No changes`.

## 14. Serving reproduzido e ensaio de recuperação

Na revisão `1adebdb`, o operador aprovou os quinze testes Python, o review
corrigido do plano, o apply e o gate HTTP. Em `2026-10-05T21:25:47.017356+00:00`,
a API do ensaio confirmou a v17 e a Run histórica, as mesmas métricas, o digest
do deployment/pod e o modelo carregado. O lote pareado retornou os mesmos
valores nos dois ambientes:

| Hora UTC de 04/10/2026 | FC | MW |
|---|---|---|
| 10:00 | 0.06623484449484646 | 780.6637476696087 |
| 11:00 | 0.056608779451567344 | 667.2080572500081 |
| 12:00 | 0.05344438716699102 | 629.9115804663062 |

A diferença máxima foi zero em FC e MW. O plano após o apply retornou
`No changes`, preservando `api.enabled=true` e chart `0.1.4`.
Payload, respostas e identidade estão em
[`evidence/serving_reproduction_2026-10-05.json`](evidence/serving_reproduction_2026-10-05.json).
Isso comprova equivalência do serving nesse lote sintético, não precisão em
ground truth ou generalização de novas versões da aplicação.

### Recuperação declarativa de falha de pull

O Terraform passa a aceitar `api_digest` e `deployment_timeout_seconds`, com
padrões iguais ao digest validado e 600s. Digest incompleto e timeout fora de
60–900s são recusados. Os overrides finais continuam mantendo contexto,
namespace, release e estágio explícitos. A intenção local persiste digest e
timeout para que os planos seguintes usem a mesma configuração.

```bash
poetry run python -m unittest discover -s tests/infra -p test_repro_serving.py -v
.repro/bin/terraform -chdir=terraform/environments/local test
make repro-recovery-test
```

Esse último target **provoca uma falha intencional somente no ensaio**:

1. Valida novamente o serving atual e salva um checkpoint privado de imagem,
   configuração, identidade do modelo e UIDs dos dois PVCs Bound.
2. Planeja a mesma release com `sha256:` seguido de 64 zeros e timeout Helm de
   60s. Trata-se de uma referência reservada sem artefato publicado, não de uma
   nova versão funcional. Recusa mudanças de chart, hash do chart ou inputs.
3. Aplica a configuração, espera erro e consulta os pods para confirmar
   `ErrImagePull` ou `ImagePullBackOff`. Não publica mensagens brutas de erro.
4. Após tentar essa atualização, executa o rollback declarativo em `finally`:
   restaura a intenção do checkpoint, gera um novo plano e aplica pelo Terraform.
   Não usa `helm rollback`, `kubectl set image`, substituição ou destruição.
5. Repete o gate de digest, saúde, identidade e inferência pareada; verifica os
   mesmos UIDs de PVC e exige plano posterior `No changes`.

O provider Helm `3.3.0` observa o status da release no refresh e planeja o estado
`deployed`; o roteiro também suporta o caso de uma tentativa falhada não ter
gravado o novo digest no state. O rollback passa pela mesma validação de
escopo e só permite alterações de digest/timeout e recuperação do status.
Código de referência:
[`resource_helm_release.go` em v3.3.0](https://github.com/hashicorp/terraform-provider-helm/blob/v3.3.0/helm/resource_helm_release.go).

O relatório final fica em `.repro/recovery-validation.json`. O diagnóstico
sanitizado fica em `.repro/recovery-failure.json`; o log privado da tentativa
falhada é preservado em `.repro/recovery-failed-apply.log`. A confirmação de
falha tem prazo Helm de 60s, mas o comando completo inclui consultas, novos
planos, recuperação e HTTP; não se promete duração total de 60s.

Se a recuperação não terminar, o checkpoint permanece disponível:

```bash
make repro-rollback
```

Uma falha diferente da esperada continua sendo erro mesmo após recuperar o
checkpoint. Se o planejamento falhar antes de tentar o apply, nenhum workload
é alterado. Não rodar planos/applies concorrentes durante o ensaio.
Os targets separados `repro-checkpoint`, `repro-failure-plan` e
`repro-serving-apply` existem para inspeção por etapas; `repro-rollback` usa o
checkpoint já salvo, sem exigir que a imagem falhada esteja saudável.

Para uma futura imagem funcional já publicada, há `make repro-release-plan
REPRO_API_DIGEST=sha256:...`, seguido de review e `make repro-serving-apply`.
O gate valida o digest selecionado, preservando o protocolo da v17 restaurada.
O exercício com digest indisponível demonstra detecção e recuperação de falha
de pull; não substitui a validação de upgrade funcional, migração de dados ou
rollback para outro modelo. Não há restauro do banco nessa recuperação.

Vinte e dois testes Python passaram nesta revisão, incluindo checkpoint,
escopo limitado a imagem, recuperação após erro esperado/inesperado, preservação
do diagnóstico e plano sem mudanças. A configuração Terraform tem cinco testes
com mocks. O operador aprovou os 22 testes Python, os cinco testes Terraform
e o ensaio real na revisão `ca7830b`, registrado na seção 15. O runtime MLflow
e o blueprint AWS continuam como frentes posteriores.

## 15. Recuperação real aprovada

Na revisão `ca7830b`, o operador executou `make repro-recovery-test` no KinD
separado. O checkpoint passou no gate em `2026-10-05T21:50:41.313311+00:00`.
A atualização para o digest indisponível falhou e o script confirmou
`ErrImagePull`. Em seguida, restaurou a configuração do checkpoint por um novo
plano e apply Terraform. Os dois planos alteraram somente a mesma release:
zero recursos adicionados, uma atualização e zero destruídos. O chart `0.1.4`,
seu hash e os inputs Helm foram preservados.

| Verificação após recuperação | Resultado observado |
|---|---|
| Imagem da API | Digest validado `sha256:95208ab282e24014a81f60d1e3eac01b8db36e40f05ae25e9f168264e4b7c188` |
| Saúde e identidade | Modelo carregado, champion v17 e Run `1d13a61244c54f06aa70f43a9993ea37` |
| Inferência pareada | Três previsões iguais à origem; diferença máxima zero em FC e MW |
| Persistência | UIDs dos dois PVCs preservados |
| Plano posterior | `No changes`, com API habilitada e timeout de 600s |
| Testes executados pelo operador | 22 Python e cinco Terraform com mocks aprovados, zero falhas |

O gate após o rollback passou em `2026-10-05T21:51:52.298638+00:00`; o resumo
final foi registrado em `2026-10-05T21:51:53.694528+00:00`. A evidência sanitizada
está em
[`evidence/deployment_recovery_2026-10-05.json`](evidence/deployment_recovery_2026-10-05.json).
Ela registra a execução real, separada das saídas simuladas dos testes.

Esse marco fecha a recuperação de falha de pull no ensaio. Não demonstra
upgrade de uma nova versão funcional, migração de dados, rollback de modelo
ou disponibilidade contínua durante a tentativa. Não foi necessário restaurar
o banco nem promover outra versão do modelo.

A próxima entrega é empacotar as dependências do servidor MLflow em uma imagem
própria, removendo a instalação via pip em cada startup. O digest construído
precisará ser validado no ensaio com Registry, acesso aos artefatos e o mesmo
gate de serving. Depois, completar a configuração dos processos locais e o
blueprint AWS com validações estáticas e mocks, sem provisionamento pago.

## Referências do projeto

- [Operação atual](OPERATIONS.md)
- [Infraestrutura e limites de persistência validados](INFRASTRUCTURE.md)
- [Contrato e governança do modelo](MODEL_CARD.md)
