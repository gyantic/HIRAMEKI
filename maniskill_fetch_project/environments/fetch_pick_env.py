"""
Fetchロボットのピッキングタスク環境

このモジュールは、Fetchロボットを使用したピッキングタスクを実装します。
"""

import numpy as np
from .fetch_env import FetchBaseEnv


class FetchPickEnv(FetchBaseEnv):
    """
    Fetchロボットのピッキングタスク環境

    この環境では、Fetchロボットがテーブル上のオブジェクトを掴むタスクを学習します。
    """

    def __init__(self, obs_mode="rgbd", control_mode="pd_ee_delta_pose", **kwargs):
        """
        Args:
            obs_mode: 観測モード
            control_mode: 制御モード
            **kwargs: その他の引数
        """
        super().__init__(obs_mode=obs_mode, control_mode=control_mode, **kwargs)

        # タスク固有のセットアップ
        self._setup_pick_task()

    def _setup_pick_task(self):
        """ピッキングタスクのセットアップ"""
        # テーブルの作成
        self._create_table()

        # オブジェクトの作成
        self._create_object()

        # カメラの設定
        self._setup_camera()

    def _create_table(self):
        """テーブルを作成"""
        # テーブルのアセットを読み込み
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.5, 0.5, 0.02])
        builder.add_box_visual(half_size=[0.5, 0.5, 0.02], color=[0.8, 0.8, 0.8])

        self.table = builder.build_static("table")

        # テーブルの位置を設定
        table_pose = [0, 0, 0]  # 原点に配置
        self.table.set_pose(table_pose)

    def _create_object(self):
        """ピッキング対象のオブジェクトを作成"""
        # キューブを作成
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1, 0, 0])

        self.object = builder.build("object")

        # オブジェクトの初期位置を設定（テーブルの上）
        object_pose = [0.5, 0, 0.05]  # テーブルの上に配置
        self.object.set_pose(object_pose)

    def _setup_camera(self):
        """カメラを設定"""
        # カメラの位置と向きを設定
        camera_pose = [0, -1.5, 1.5]  # ロボットの前方上方
        camera_target = [0, 0, 0]  # カメラが向く方向
        self.scene.add_camera("base_camera", camera_pose, camera_target)

    def _compute_reward(self):
        """報酬を計算"""
        # ロボットのエンドエフェクタ位置
        ee_pos = self.robot.get_ee_pose()[0]

        # オブジェクトの位置
        object_pos = self.object.get_pose()[0]

        # 距離を計算
        distance = np.linalg.norm(ee_pos - object_pos)

        # 報酬の計算（距離が近いほど高い報酬）
        reward = -distance

        # 掴み成功の判定
        if self._check_grasp_success():
            reward += 10.0  # 成功ボーナス

        return reward

    def _check_grasp_success(self):
        """掴み成功をチェック"""
        # エンドエフェクタがオブジェクトに十分近いかチェック
        ee_pos = self.robot.get_ee_pose()[0]
        object_pos = self.object.get_pose()[0]

        distance = np.linalg.norm(ee_pos - object_pos)
        return distance < 0.05  # 5cm以内

    def _check_termination(self):
        """終了条件をチェック"""
        # 掴み成功で終了
        if self._check_grasp_success():
            return True

        # 最大ステップ数で終了
        if self.episode_step >= 200:
            return True

        return False

    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed, options=options)

        # オブジェクトの位置をランダムに変更
        self._randomize_object_position()

        return self.get_obs(), {}

    def _randomize_object_position(self):
        """オブジェクトの位置をランダムに変更"""
        # テーブル上のランダムな位置
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.05  # テーブルの高さ

        object_pose = [x, y, z]
        self.object.set_pose(object_pose)

    def get_obs(self):
        """観測を取得"""
        obs = super().get_obs()

        # 画像観測を追加
        if hasattr(self, 'scene') and hasattr(self.scene, 'cameras'):
            images = self.render()
            if isinstance(images, dict) and "base_camera" in images:
                obs["image"] = images["base_camera"]
            elif isinstance(images, np.ndarray):
                obs["image"] = images
            else:
                # デフォルトの画像を作成
                obs["image"] = np.zeros((64, 64, 3), dtype=np.uint8)

        return obs
