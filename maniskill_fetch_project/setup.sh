#!/bin/bash

echo "Setting up Maniskill Fetch Robot with DreamerV3 project..."

# 依存関係のインストール
echo "Installing dependencies..."
pip install -r requirements.txt

# ディレクトリ構造の作成
echo "Creating directory structure..."
mkdir -p environments
mkdir -p training/configs
mkdir -p training/utils
mkdir -p inference/utils
mkdir -p data/checkpoints
mkdir -p data/logs
mkdir -p data/results
mkdir -p scripts

# __init__.pyファイルの作成
touch environments/__init__.py
touch training/__init__.py
touch training/utils/__init__.py
touch inference/__init__.py
touch inference/utils/__init__.py

# 実行権限の付与
chmod +x scripts/*.sh 2>/dev/null || true

echo "Setup completed!"
echo ""
echo "Next steps:"
echo "1. Create your custom Fetch environment in environments/"
echo "2. Configure training parameters in training/configs/"
echo "3. Run training with: python training/train_fetch.py --config training/configs/fetch_pick.yaml"
