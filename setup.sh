#!/bin/bash
# Setup script to install system dependencies required for the project

set -e

echo "🔍 Checking system dependencies..."

# Check if python3.11-dev is installed
if ! dpkg -l | grep -q "python3.11-dev"; then
    echo "❌ python3.11-dev is not installed"
    echo "📦 Installing python3.11-dev..."
    sudo apt-get update
    sudo apt-get install -y python3.11-dev build-essential
else
    echo "✅ python3.11-dev is already installed"
fi

# Check if Python.h exists
if [ ! -f "/usr/include/python3.11/Python.h" ]; then
    echo "❌ Python.h not found in /usr/include/python3.11/"
    echo "📦 Trying to install python3.11-dev again..."
    sudo apt-get install -y python3.11-dev
else
    echo "✅ Python.h found"
fi

echo "✅ System dependencies verified!"
echo ""
echo "📦 Installing project dependencies with uv..."
cd "$(dirname "$0")"
uv sync

echo ""
echo "✅ Setup completed!"

