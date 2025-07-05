#!/usr/bin/env python3
"""
Maniskill環境でのDreamerV3使用例

このスクリプトは、Maniskill環境でDreamerV3を学習する方法を示します。
"""

import argparse
import pathlib
import sys

# 現在のディレクトリをパスに追加
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamer import main
import tools


def create_maniskill_config():
    """Maniskill環境用の設定を作成"""
    config = argparse.Namespace()

    # 基本設定
    config.logdir = "./logdir/maniskill_example"
    config.seed = 0
    config.steps = 1e6
    config.eval_every = 1e4
    config.eval_episode_num = 10
    config.log_every = 1e4
    config.reset_every = 0
    config.device = "cuda:0"
    config.compile = True
    config.precision = 32
    config.debug = False
    config.video_pred_log = True

    # 環境設定
    config.task = "maniskill_PickCube-v1"  # 環境名
    config.size = [64, 64]  # 画像サイズ
    config.envs = 4  # 並列環境数
    config.action_repeat = 1
    config.time_limit = 200  # Maniskillの標準的なエピソード長
    config.camera_name = "base_camera"

    # モデル設定
    config.dyn_hidden = 512
    config.dyn_deter = 512
    config.dyn_stoch = 32
    config.dyn_discrete = 32
    config.dyn_rec_depth = 1
    config.dyn_mean_act = "none"
    config.dyn_std_act = "sigmoid2"
    config.dyn_min_std = 0.1
    config.grad_heads = ["decoder", "reward", "cont"]
    config.units = 512
    config.act = "SiLU"
    config.norm = True

    # エンコーダー設定
    config.encoder = {
        "mlp_keys": "$^",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 32,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 5,
        "mlp_units": 1024,
        "symlog_inputs": True
    }

    # デコーダー設定
    config.decoder = {
        "mlp_keys": "$^",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 32,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 5,
        "mlp_units": 1024,
        "cnn_sigmoid": False,
        "image_dist": "mse",
        "vector_dist": "symlog_mse",
        "outscale": 1.0
    }

    # アクター設定
    config.actor = {
        "layers": 5,
        "dist": "normal",
        "entropy": 3e-4,
        "unimix_ratio": 0.01,
        "std": "learned",
        "min_std": 0.1,
        "max_std": 1.0,
        "temp": 0.1,
        "lr": 3e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 1.0
    }

    # クリティック設定
    config.critic = {
        "layers": 5,
        "dist": "symlog_disc",
        "slow_target": True,
        "slow_target_update": 1,
        "slow_target_fraction": 0.02,
        "lr": 3e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 0.0
    }

    # 報酬ヘッド設定
    config.reward_head = {
        "layers": 5,
        "dist": "symlog_disc",
        "loss_scale": 1.0,
        "outscale": 0.0
    }

    # 継続ヘッド設定
    config.cont_head = {
        "layers": 5,
        "loss_scale": 1.0,
        "outscale": 1.0
    }

    # 学習設定
    config.dyn_scale = 0.5
    config.rep_scale = 0.1
    config.kl_free = 1.0
    config.weight_decay = 0.0
    config.unimix_ratio = 0.01
    config.initial = "learned"
    config.batch_size = 16
    config.batch_length = 64
    config.train_ratio = 512
    config.pretrain = 100
    config.model_lr = 1e-4
    config.opt_eps = 1e-8
    config.grad_clip = 1000
    config.dataset_size = 1000000
    config.opt = "adam"

    # 行動設定
    config.discount = 0.997
    config.discount_lambda = 0.95
    config.imag_horizon = 15
    config.imag_gradient = "dynamics"
    config.imag_gradient_mix = 0.0
    config.eval_state_mean = False

    # 探索設定
    config.expl_behavior = "greedy"
    config.expl_until = 0
    config.expl_extr_scale = 0.0
    config.expl_intr_scale = 1.0
    config.disag_target = "stoch"
    config.disag_log = True
    config.disag_models = 10
    config.disag_offset = 1
    config.disag_layers = 4
    config.disag_units = 400
    config.disag_action_cond = False

    # その他の設定
    config.grayscale = False
    config.prefill = 2500
    config.reward_EMA = True
    config.parallel = False
    config.traindir = None
    config.evaldir = None
    config.offline_traindir = ""
    config.offline_evaldir = ""
    config.deterministic_run = False

    return config


def test_maniskill_env():
    """Maniskill環境のテスト"""
    print("Testing Maniskill environment...")

    try:
        import envs.maniskill as maniskill

        # 環境の作成
        env = maniskill.make_maniskill_env(
            "PickCube-v1",
            action_repeat=1,
            size=(64, 64),
            camera_name="base_camera",
            seed=0
        )

        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # テスト実行
        obs = env.reset()
        print(f"Initial observation keys: {list(obs.keys())}")
        print(f"Image shape: {obs['image'].shape}")

        # ランダム行動で数ステップ実行
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: reward={reward:.3f}, done={done}")
            if done:
                obs = env.reset()

        env.close()
        print("Maniskill environment test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing Maniskill environment: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Maniskill環境でのDreamerV3学習")
    parser.add_argument("--test", action="store_true", help="環境のテストのみ実行")
    parser.add_argument("--env", type=str, default="PickCube-v1", help="使用する環境名")
    parser.add_argument("--steps", type=float, default=1e6, help="学習ステップ数")
    parser.add_argument("--logdir", type=str, default="./logdir/maniskill_example", help="ログディレクトリ")

    args = parser.parse_args()

    if args.test:
        # 環境のテストのみ実行
        success = test_maniskill_env()
        if success:
            print("\n環境のテストが成功しました。学習を開始できます。")
        else:
            print("\n環境のテストに失敗しました。Maniskill2のインストールを確認してください。")
        return

    # 設定の作成
    config = create_maniskill_config()

    # 引数で上書き
    config.task = f"maniskill_{args.env}"
    config.steps = args.steps
    config.logdir = args.logdir

    print(f"Starting DreamerV3 training on Maniskill environment: {args.env}")
    print(f"Log directory: {config.logdir}")
    print(f"Training steps: {config.steps}")

    # 学習開始
    main(config)


if __name__ == "__main__":
    main()
