# Locomotion

UnitreeGo2の強化学習による移動制御プロジェクト

## プロジェクト概要

物理シミュレータ[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)を使用して、UnitreeGo2に様々な移動タスクを学習させるプロジェクトです。

### 実装済み機能
- ✅ 平地歩行（Walking）
- ✅ ジャンプ（Jump）
- ✅ 走行（Running）
- ✅ バックフリップデモ（Backflip Demo）

### 開発中機能（ブランチで開発中）
- 🚧 段差地形歩行（Terrain Walking）- `feature/terrain-walking`ブランチ
- 🚧 バックフリップの学習（Backflip Training）- `feature/backflip`ブランチ

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

### 学習進捗の確認（全機能共通）
訓練中または訓練後にTensorBoardで学習進捗を確認できます：
```bash
tensorboard --logdir logs
```
ブラウザで `http://localhost:6006` を開いてください。

### 歩行（Walking）

**既存モデルを使用する場合:**
```bash
python go2_eval.py -e go2-walking --ckpt 100
```

**自分で学習させる場合:**
```bash
python go2_train.py -e go2-walking --max_iterations 101
python go2_eval.py -e go2-walking --ckpt 100
```

### ジャンプ（Jump）

**既存モデルを使用する場合:**
```bash
python go2_eval.py -e go2-jump --ckpt 100
```

**自分で学習させる場合:**
```bash
python go2_train.py -e go2-jump --max_iterations 101
python go2_eval.py -e go2-jump --ckpt 100
```

### 走行（Running）

四足歩行ロボット研究に基づいたカスタム報酬関数を実装:
- **Forward Distance** (10.0): 前進距離を直接報酬化 ★最重要
- **Diagonal Gait** (0.5): 対角歩容（トロット）の奨励
- **Aligned Hips** (0.3): ヒップ関節の整列
- **Straight Line** (0.5): 直進性の維持
- **Foot Clearance** (0.2): 足の持ち上げ
- **Energy Efficiency** (-0.001): エネルギー効率（ペナルティ）

<details>
<summary>報酬スケールの詳細</summary>

#### 基本的な報酬（歩行から継承・調整）
| 報酬名 | スケール | 歩行時 | 説明 |
|--------|----------|--------|------|
| tracking_lin_vel | 1.5 | 1.0 | 目標速度(1.5-2.0 m/s)への追従 |
| tracking_ang_vel | 0.2 | 0.2 | 角速度の追従 |
| lin_vel_z | -0.5 | -1.0 | Z軸速度ペナルティ（走行時の上下動を許容） |
| base_height | -30.0 | -50.0 | 高さ乖離ペナルティ（ダイナミックな動きを許容） |
| action_rate | -0.002 | -0.005 | アクション変化ペナルティ（動的な関節動作を許可） |
| similar_to_default | -0.05 | -0.1 | デフォルト姿勢乖離ペナルティ（大きな動作を許容） |

#### カスタム報酬（走行特化）
| 報酬名 | スケール | 説明 |
|--------|----------|------|
| **forward_distance** | **10.0** | **実際に前進した距離を報酬化（最重要）** |
| diagonal_gait | 0.5 | 対角線上の脚（FR-RL, FL-RR）の同期動作 |
| aligned_hips | 0.3 | 4つのヒップ関節の角度の一貫性 |
| straight_line | 0.5 | Y軸方向のずれを抑制 |
| foot_clearance | 0.2 | 膝関節の角速度（ダイナミックな足運び） |
| energy_efficiency | -0.001 | トルクの二乗和を最小化 |

**毎ステップの最終報酬** = 全報酬関数の重み付き合計

</details>

**既存モデルを使用する場合:**
```bash
python go2_running_eval.py -e go2-running --ckpt 100
```

**自分で学習させる場合:**
```bash
python go2_running_train.py -e go2-running --max_iterations 500
python go2_running_eval.py -e go2-running --ckpt 100
```

**途中から学習を再開する場合:**
```bash
# model_100.ptから500イテレーションまで続ける
python go2_running_train.py -e go2-running --resume --ckpt 100 --max_iterations 500

# model_200.ptから1000イテレーションまで
python go2_running_train.py -e go2-running --resume --ckpt 200 --max_iterations 1000
```

### バックフリップ（Backflip）

**既存モデルを使用する場合:**
```bash
python go2_backflip.py -e single   # シングルバックフリップ
python go2_backflip.py -e double   # ダブルバックフリップ
```

---

## 開発中機能

以下の機能は現在開発中のため、対応するブランチに切り替えて使用してください。

### バックフリップの学習（Backflip Training）🚧

バックフリップのデモは上記のQuick Startから実行できますが、自分で学習させる機能はまだ開発中です。

**ブランチ切り替え:**
```bash
git switch feature/backflip
cd backflip
```

**訓練（開発中）:**
```bash
python train_backflip.py -e go2-backflip --max_iterations 101
python eval_backflip.py -e go2-backflip --ckpt 100
```

### 段差地形歩行（Terrain Walking）🚧

**ブランチ切り替え:**
```bash
git switch feature/terrain-walking
cd terrain
```

**追加の依存関係:**
```bash
pip install pygame
```

**訓練（開発中）:**
```bash
python go2_terrain_train.py -e go2-terrain-walking --max_iterations 101
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
├── go2_running_env.py      # 走行専用環境クラス
├── go2_running_train.py    # 走行訓練スクリプト
├── go2_running_eval.py     # 走行評価スクリプト
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
python go2_running_train.py -e go2-running -B 2048  # 走行の場合
```

### TensorBoardが起動しない
```bash
pip install --upgrade tensorboard
```

### 途中から学習を再開したい
`--resume`フラグと`--ckpt`オプションを使用します：
```bash
# 例: model_100.ptから続きを学習
python go2_running_train.py -e go2-running --resume --ckpt 100 --max_iterations 500
```
注意: 歩行・ジャンプ用の`go2_train.py`には現在resume機能はありません。

## ライセンス

教育・研究目的で使用してください。

## 参考資料

- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [RSL-RL Library](https://github.com/leggedrobotics/rsl_rl)
- [Genesis Backflip](https://github.com/ziyanx02/Genesis-backflip)
- [Go2 Tasks](https://www.sharwinpatil.info/posts/go2-tasks/)
