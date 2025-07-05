#!/usr/bin/env python3
"""
maniskill_fetch_projectのカスタム環境をテストするスクリプト
"""

import sys
import pathlib
import numpy as np

# maniskill_fetch_projectのパスを追加
sys.path.append(str(pathlib.Path(__file__).parent.parent / "maniskill_fetch_project"))

def test_fetch_pick_env():
    """FetchPickEnvのテスト"""
    print("Testing FetchPickEnv...")

    try:
        from environments.fetch_pick_env import FetchPickEnv
        from environments.fetch_env import make_fetch_env

        # 環境の作成
        env = make_fetch_env(
            FetchPickEnv,
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
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: reward={reward:.3f}, done={done}")
            if done:
                obs = env.reset()
                print("Episode ended, resetting...")

        env.close()
        print("FetchPickEnv test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing FetchPickEnv: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dreamer_integration():
    """DreamerV3との統合テスト"""
    print("Testing DreamerV3 integration...")

    try:
        import envs.maniskill as maniskill

        # カスタム環境の作成
        env = maniskill.make_maniskill_env(
            "FetchPick-v1",
            action_repeat=1,
            size=(64, 64),
            camera_name="base_camera",
            seed=0
        )

        print(f"DreamerV3 observation space: {env.observation_space}")
        print(f"DreamerV3 action space: {env.action_space}")

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
        print("DreamerV3 integration test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing DreamerV3 integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== FetchPickEnv Test ===")
    success1 = test_fetch_pick_env()

    print("\n=== DreamerV3 Integration Test ===")
    success2 = test_dreamer_integration()

    if success1 and success2:
        print("\n✅ All tests passed! You can now train DreamerV3 on the custom environment.")
        print("\nTo start training, run:")
        print("cd dreamerv3-torch")
        print("python dreamer.py --configs maniskill_fetch")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
