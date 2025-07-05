# Maniskill Fetch Robot with DreamerV3

このプロジェクトは、Maniskill2のカスタム環境でFetchロボットを使用し、DreamerV3で学習・推論を行うためのプロジェクトです。

## プロジェクト構造

```
maniskill_fetch_project/
├── README.md                    # このファイル
├── requirements.txt             # 依存関係
├── setup.sh                     # セットアップスクリプト
├──
├── environments/                # カスタム環境
│   ├── __init__.py
│   ├── fetch_env.py            # Fetchロボット基本環境
│   ├── fetch_pick_env.py       # ピッキングタスク
│   ├── fetch_place_env.py      # プレイスメントタスク
│   └── fetch_assembly_env.py   # 組み立てタスク
│
├── training/                    # 学習関連
│   ├── __init__.py
│   ├── train_fetch.py          # 学習スクリプト
│   ├── configs/                # 設定ファイル
│   │   ├── fetch_pick.yaml
│   │   ├── fetch_place.yaml
│   │   └── fetch_assembly.yaml
│   └── utils/                  # 学習用ユーティリティ
│       ├── __init__.py
│       ├── logger.py
│       └── checkpoint.py
│
├── inference/                   # 推論関連
│   ├── __init__.py
│   ├── evaluate_fetch.py       # 評価スクリプト
│   ├── visualize_fetch.py      # 可視化スクリプト
│   └── utils/                  # 推論用ユーティリティ
│       ├── __init__.py
│       └── renderer.py
│
├── data/                       # データ保存
│   ├── checkpoints/           # 学習済みモデル
│   ├── logs/                  # 学習ログ
│   └── results/               # 推論結果
│
└── scripts/                    # 実行スクリプト
    ├── train_pick.sh          # ピッキング学習
    ├── train_place.sh         # プレイスメント学習
    ├── evaluate_pick.sh       # ピッキング評価
    └── visualize_results.sh   # 結果可視化
```

## セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt

# セットアップスクリプトの実行
./setup.sh
```

## 使用方法

### 学習

```bash
# ピッキングタスクの学習
python training/train_fetch.py --config configs/fetch_pick.yaml

# プレイスメントタスクの学習
python training/train_fetch.py --config configs/fetch_place.yaml
```

### 推論・評価

```bash
# 学習済みモデルの評価
python inference/evaluate_fetch.py --checkpoint data/checkpoints/fetch_pick_latest.pt

# 結果の可視化
python inference/visualize_fetch.py --checkpoint data/checkpoints/fetch_pick_latest.pt
```

## カスタム環境の追加

新しいタスクを追加する場合：

1. `environments/`に新しい環境クラスを作成
2. `training/configs/`に設定ファイルを追加
3. `scripts/`に実行スクリプトを追加

## 注意事項

- DreamerV3の実装は`../dreamerv3-torch`を参照
- Maniskill2のFetchロボットアセットが必要
- GPU使用を推奨（CPUでも動作可能）
