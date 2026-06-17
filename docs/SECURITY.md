# Modelo de Segurança — loan-risk-analyzer

Este documento descreve as decisões de segurança da Entrega 1, com foco em dois
riscos: **isolamento da execução** (o script não deve acessar o ambiente local)
e **supply chain** (dependências comprometidas não devem comprometer o host).

## 1. Modelo de ameaças (resumido)

| Ativo | Ameaça | Mitigação |
|-------|--------|-----------|
| Dados de crédito do solicitante (CSV) | Exfiltração por dependência maliciosa | Container **sem rede** (`network_mode: none`) |
| Sistema de arquivos do host | Escrita/leitura arbitrária pelo script | Raiz **read-only** + apenas volumes montados |
| Dataset original (`loan_data.csv`) | Adulteração | Volume de entrada montado **somente leitura** |
| Cadeia de dependências | Pacote/versão adulterada (typosquatting, conta comprometida) | **Hashes fixados** + `--require-hashes` + `--no-deps` |
| Kernel do host | Escalonamento de privilégio | `cap_drop: ALL`, `no-new-privileges`, **non-root** |

> Premissa: a entrada (CSV) é **não confiável**. O parser usa apenas a
> biblioteca padrão (`csv`) e trata o conteúdo como dados, sem `eval`/execução.

## 2. Isolamento da execução

A aplicação roda em um container Docker configurado para o **mínimo
privilégio**. As diretivas em `docker-compose.yml`:

- **`network_mode: none`** — o container não tem interface de rede. Mesmo que
  uma dependência tente "telefonar para casa" ou exfiltrar dados, não há rota
  de saída. O pipeline é 100% local.
- **`read_only: true`** — o sistema de arquivos raiz é imutável. A única área
  gravável é o volume `data/output` e um `tmpfs` de 64 MB em `/tmp`.
- **Volume de entrada `read_only`** — `data/loan_data.csv` é montado como
  somente leitura; o script não pode alterar o dataset original.
- **`cap_drop: ALL` + `no-new-privileges:true`** — remove todas as
  capabilities do kernel e impede ganho de privilégio em tempo de execução.
- **Usuário `non-root`** — definido no `Dockerfile` (`USER app`) e reforçado no
  compose (`user: "app"`).
- **Limites de recurso** — `pids_limit`, `mem_limit` e `cpus` contêm consumo
  anômalo (loops, fork bombs, mineração).

Resultado: o script "vê" somente os volumes de dados montados explicitamente
e não tem como tocar no restante da máquina nem na rede.

## 3. Defesa de supply chain

O risco de supply chain é tratado em camadas:

1. **Superfície mínima.** `requirements.in` lista o conjunto mínimo de pacotes.
   Menos dependências = menos vetores de ataque.
2. **Versões fixas + hashes.** `requirements.txt` é gerado com
   `pip-compile --generate-hashes`. Cada artefato tem seu hash SHA-256.
3. **Instalação verificada.** O `Dockerfile` instala com
   `pip install --require-hashes --no-deps`:
   - `--require-hashes` faz o pip **abortar** se qualquer arquivo baixado
     divergir do hash registrado (detecta adulteração no registro/mirror);
   - `--no-deps` garante que **nada além** do que foi explicitamente travado
     seja instalado (impede injeção de dependência transitiva).
4. **Build multi-stage.** A imagem final não contém compiladores nem o cache de
   build, reduzindo a superfície da imagem entregue.
5. **Auditoria contínua.** `make audit` roda `pip-audit` sobre o
   `requirements.txt`, comparando as versões travadas com bases de
   vulnerabilidades conhecidas (CVE/OSV).

### Como atualizar dependências com segurança

```bash
# 1. Editar requirements.in (apenas o necessário)
# 2. Regerar o lock com hashes
make lock
# 3. Auditar antes de commitar
make audit
```

## 4. Boas práticas de dados

- O dataset `loan_data.csv` é montado como volume **somente leitura** e não é
  copiado para a imagem (ver `.dockerignore`).
- Nenhum segredo/credencial é embutido na imagem ou no código.
- Caminhos de entrada/saída vêm de variáveis de ambiente, nunca hardcoded para
  o host.

## 5. Limitações conhecidas

- O isolamento depende da configuração do `docker-compose.yml`; executar a
  imagem com `docker run` sem essas flags reduz as garantias. Use sempre
  `make run` / `docker compose run`.
- `--require-hashes` protege a integridade, não a *intenção*: um pacote
  legítimo porém vulnerável só é detectado pela etapa de `pip-audit`.
