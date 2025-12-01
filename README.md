# SPLADE-PT-BR

## Instalação

### Pré-requisitos

Este projeto requer os headers de desenvolvimento do Python 3.11 para compilar o pacote `pytrec-eval`. 

**Instale as dependências do sistema:**

```bash
sudo apt-get update
sudo apt-get install -y python3.11-dev build-essential
```

### Setup do Projeto

1. **Instalação automática (recomendado):**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Instalação manual:**
   ```bash
   uv sync
   ```

### Verificação

Para verificar se tudo está instalado corretamente:

```bash
# Verificar pytrec-eval
python3.11 -c "import pytrec_eval; print('✅ pytrec-eval instalado com sucesso!')"

# Verificar dependências principais
python3.11 -c "import torch; import transformers; import omegaconf; print('✅ Todas as dependências principais instaladas!')"
```

### Dependências Principais

O projeto requer as seguintes dependências Python:
- `torch` (PyTorch) - Para treinamento de modelos
- `transformers` - Para modelos de linguagem
- `hydra-core` - Para gerenciamento de configurações
- `omegaconf` - Para configurações (instalado com hydra-core)
- `pytrec-eval` - Para avaliação de recuperação de informação
- `numpy`, `scipy`, `scikit-learn` - Para processamento numérico

## Estrutura do Projeto

- `.venv/` - Ambiente virtual Python
- `pyproject.toml` - Configuração do projeto e dependências
- `SPLADE_v2_PTBR_treinamento.ipynb` - Notebook de treinamento