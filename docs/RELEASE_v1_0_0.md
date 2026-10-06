# v1.0.0 — IaC e deployment reproduzível

Notas para a publicação do marco de portfólio. A implementação foi integrada à
`main` pelo [PR #4](https://github.com/wanderson42/renewable-energy-mlops/pull/4),
commit `2b07c406bde02fd26560b1565bf10acd4da64eca`, em 05/10/2026 no fuso
America/Belem. Publicar uma etiqueta/release distribui esse marco; estas notas
não constituem evidência de que a etiqueta já foi publicada.

## Resultado

O projeto passa de uma instalação local preparada manualmente para um ensaio
reproduzível em outro KinD: configuração versionada, release Terraform/Helm,
restauração de PostgreSQL/S3, verificação do modelo servido e acesso por
clientes isolados. O blueprint AWS declara uma possível infraestrutura cloud,
validada gratuitamente sem provisionamento.

| Marco | Entrega preservada |
|---|---|
| v0.1 | Sistema MLOps funcional, dados/treino/serving e evidências E2E históricas |
| v0.2 | Experimentos com snapshot e OOT controlados; ablações e janelas temporais |
| v0.3 | Governança same-OOT, champion v17, monitoring real e caminhos Client/E2E |
| Otimização Docker | Redução local de 67,46%; publicação e inferência por digest |
| v1.0 | IaC, reprodução local, recuperação do estado e blueprint AWS |

## Entrega local exercitada

- KinD separado, kubeconfig/contexto/namespace explícitos e imagens por digest.
- Release gerenciada por Terraform, com chart Helm e plano final sem mudanças.
- Snapshot PostgreSQL/RustFS restaurado; 251 objetos S3 verificados por SHA-256.
- Champion v17 e Run `1d13a61244c54f06aa70f43a9993ea37` preservados; três
  previsões pareadas em FC/MW com diferença máxima zero.
- Falha controlada de pull da imagem API e recuperação declarativa, com
  identidades dos PVCs preservadas.
- Runtime MLflow empacotado, publicado e adotado por digest, sem instalação de
  dependências no startup; Registry e leitura/hash de artefato verificados.
- Cliente Poetry com Prefect/Streamlit, portas loopback próprias e estado
  Prefect isolado. Gate SDK/HTTP e recuperação manual da interface confirmados.

## Blueprint AWS

Terraform declara VPC em duas AZs, EKS privado por padrão, buckets S3 privados,
RDS PostgreSQL Multi-AZ e IAM/IRSA com permissões por workload. Foram executados
validação de configuração, sete testes com mocks, TFLint e vinte controles
Checkov selecionados. Não houve plano autenticado, apply ou deployment AWS.
A topologia e as integrações futuras estão em [AWS_BLUEPRINT.md](AWS_BLUEPRINT.md).

## Validação e proveniência

Os quatro workflows da `main` passaram após o merge da implementação:

| Verificação | Resultado |
|---|---|
| Suíte Python/Tox | 225 passed, 1 skipped, 2 warnings; 56 subtests passed; py314 OK |
| Terraform local | 6 testes com mocks aprovados; inputs reais da CLI testados separadamente |
| Terraform AWS | 7 testes com mocks aprovados; TFLint e 20 controles Checkov aprovados |
| Helm e scripts | Lint, renderização e guards de infraestrutura aprovados |
| Imagens | Build/publicação FastAPI e build/smoke/publicação MLflow aprovados |

O teste ignorado no job Python exige a CLI Terraform e passou no job Terraform
local. Subtestes e testes IaC não são somados ao total de 225. Os dois warnings
vêm do código legado Evidently/NumPy. Não foi medido percentual de cobertura.

O [recibo de consolidação](evidence/v1_consolidation_2026-10-05.json) vincula
números, jobs e revisão. Os recibos operacionais anteriores preservam a evidência
do host. A execução CI não repete o apply nem a restauração no KinD do operador.

## Limites do fechamento

O escopo de implementação do portfólio está concluído. O ambiente exercitado é
local; mocks não demonstram provisionabilidade ou disponibilidade AWS. O apply
local é iniciado pelo operador, e o Actions não implanta no KinD automaticamente.

A reprodução da identidade histórica depende do snapshot privado, ausente do
Git. O restore não constitui prova de recuperação após perda total do host nem
política de retenção externa. O banco Prefect do ensaio é novo: execução de
flows/workers, restauração do histórico e browser E2E automatizado não foram
exercitados nesta etapa.

O acompanhamento longitudinal da v0.3 continua com revisões de 7 e 14 dias
completos de geração observada e do mês fechado de outubro. A avaliação com
meteorologia observada/reanálise é distinta da validação de previsões day-ahead
efetivamente emitidas. Essas frentes não alteram o critério de fechamento da
implementação v1.0 nem disparam retreinamento automático.

## Navegação

- [README e árvore do repositório](../README.md).
- [Arquitetura local](INFRASTRUCTURE.md).
- [Runbook de reprodução](REPRODUCIBLE_DEPLOYMENT.md).
- [Notebook próprio da infraestrutura](../notebooks/operations/reproducible_deployment_v1_0.ipynb).
- [Notebook principal: sínteses da evolução](../notebooks/renewable-energy-mlops.ipynb).
