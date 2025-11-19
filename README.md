# Locomotion

Unitree Go2ロボットの強化学習による移動制御プロジェクト

## プロジェクト概要

Genesis物理シミュレータを使用して、Go2四足ロボットに様々な移動タスクを学習させるプロジェクトです。

### 実装済み機能
- ✅ 平地歩行（Walking）
- ✅ ジャンプ（Jump）

### 開発中機能（ブランチで開発中）
- 🚧 段差地形歩行（Terrain Walking）- `feature/terrain-walking`ブランチ
- 🚧 バックフリップ（Backflip）- `feature/backflip`ブランチ

## 環境構築（Mac）

### 1. Miniforgeのインストール

```bash
brew install miniforge
mamba shell init --shell zsh --root-prefix=/opt/homebrew/opt/miniforge
```

### 2. 環境セットアップ

#### 方法A: environment.yml使用版（推奨）

```bash
git clone https://github.com/tadryo/locomotion.git
cd locomotion
mamba env create -n locomotion -f environment.yml
mamba activate locomotion
```

#### 方法B: 手動インストール版

```bash
mamba create -n locomotion python=3.10 -y
mamba activate locomotion
pip install rsl-rl-lib==2.2.4 tensorboard

# Genesisのインストール
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e ".[dev]"
cd ..

# Locomotionのクローン
git clone https://github.com/tadryo/locomotion.git
cd locomotion
```

## Quick Start

### 歩行（Walking）

**既存モデルを使用する場合:**
```bash
python go2_eval.py -e go2-walking --ckpt 100
```

**自分で学習させる場合:**
```bash
# 訓練
python go2_train.py -e go2-walking --max_iterations 101

# TensorBoardで学習進捗を確認
tensorboard --logdir logs

# 評価
python go2_eval.py -e go2-walking --ckpt 100
```

### ジャンプ（Jump）

**既存モデルを使用する場合:**
```bash
python go2_eval.py -e go2-jump --ckpt 100
```

**自分で学習させる場合:**
```bash
# 訓練
python go2_train.py -e go2-jump --max_iterations 101

# TensorBoardで学習進捗を確認
tensorboard --logdir logs

# 評価
python go2_eval.py -e go2-jump --ckpt 100
```

---

## 開発中機能

以下の機能は現在開発中のため、対応するブランチに切り替えて使用してください。

### バックフリップ（Backflip）🚧

**ブランチ切り替え:**
```bash
git switch feature/backflip
```

**既存モデルを使用する場合:**
```bash
python go2_backflip.py -e single   # シングルバックフリップ
python go2_backflip.py -e double   # ダブルバックフリップ
```

**自分で学習させる場合:**
```bash
cd backflip

# 訓練
python train_backflip.py -e go2-backflip --max_iterations 101

# TensorBoardで学習進捗を確認
tensorboard --logdir logs

# 評価
python eval_backflip.py -e go2-backflip --ckpt 100
```

### 段差地形歩行（Terrain Walking）🚧

**ブランチ切り替え:**
```bash
git switch feature/terrain-walking
```

**追加の依存関係:**
```bash
pip install pygame
```

**既存モデルを使用する場合:**
```bash
cd terrain
python go2_terrain_eval.py -e go2-terrain-walking --ckpt 100
```

**自分で学習させる場合:**
```bash
cd terrain

# 訓練
python go2_terrain_train.py -e go2-terrain-walking --max_iterations 101

# TensorBoardで学習進捗を確認
tensorboard --logdir logs

# 評価
python go2_terrain_eval.py -e go2-terrain-walking --ckpt 100
```

---

## ブランチ構成

```
main (安定版)
  ├── feature/terrain-walking (段差地形歩行開発中)
  └── feature/backflip (バックフリップ開発中)
```

### mainブランチに戻る
```bash
git switch main
```

## プロジェクト構成

```
locomotion/
├── go2_env.py              # 基本環境クラス
├── go2_train.py            # 訓練スクリプト（歩行・ジャンプ）
├── go2_eval.py             # 評価スクリプト（歩行・ジャンプ）
├── go2_backflip.py         # バックフリップデモ
├── backflip/               # バックフリップ開発ディレクトリ（feature/backflipブランチ）
├── terrain/                # 地形歩行開発ディレクトリ（feature/terrain-walkingブランチ）
├── logs/                   # 訓練ログとモデル
└── environment.yml         # Python環境設定
```

## トラブルシューティング

### GPUメモリ不足
並列環境数を減らしてください：
```bash
python go2_train.py -e go2-walking -B 2048  # デフォルトは4096
```

### TensorBoardが起動しない
```bash
pip install --upgrade tensorboard
```

## ライセンス

教育・研究目的で使用してください。

## 参考資料

- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [RSL-RL Library](https://github.com/leggedrobotics/rsl_rl)
