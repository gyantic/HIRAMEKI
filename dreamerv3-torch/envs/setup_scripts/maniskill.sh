#!/bin/bash

# Maniskill環境のセットアップスクリプト

echo "Installing Maniskill2..."

# Maniskill2のインストール
pip install mani-skill2

# 依存関係のインストール
pip install sapien
pip install gymnasium

# 環境のダウンロード（オプション）
# python -m mani_skill2.utils.download_assets

echo "Maniskill2 installation completed!"
echo ""
echo "Available environments:"
echo "- PickCube-v1"
echo "- StackCube-v1"
echo "- PegInsertionSide-v1"
echo "- Assembly-v1"
echo "- PlugCharger-v1"
echo "- TurnFaucet-v1"
echo "- OpenCabinetDoor-v1"
echo "- OpenCabinetDrawer-v1"
echo "- PushChair-v1"
echo "- MoveBucket-v1"
echo ""
echo "Usage example:"
echo "python3 dreamer.py --configs maniskill --task maniskill_PickCube-v1 --logdir ./logdir/maniskill_pickcube"
