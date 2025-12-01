#!/bin/bash
# Script de setup para instalar dependências do sistema necessárias para o projeto

set -e

echo "🔍 Verificando dependências do sistema..."

# Verificar se python3.11-dev está instalado
if ! dpkg -l | grep -q "python3.11-dev"; then
    echo "❌ python3.11-dev não está instalado"
    echo "📦 Instalando python3.11-dev..."
    sudo apt-get update
    sudo apt-get install -y python3.11-dev build-essential
else
    echo "✅ python3.11-dev já está instalado"
fi

# Verificar se Python.h existe
if [ ! -f "/usr/include/python3.11/Python.h" ]; then
    echo "❌ Python.h não encontrado em /usr/include/python3.11/"
    echo "📦 Tentando instalar python3.11-dev novamente..."
    sudo apt-get install -y python3.11-dev
else
    echo "✅ Python.h encontrado"
fi

echo "✅ Dependências do sistema verificadas!"
echo ""
echo "📦 Instalando dependências do projeto com uv..."
cd "$(dirname "$0")"
uv sync

echo ""
echo "✅ Setup concluído!"

