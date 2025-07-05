import gym
import numpy as np
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import flatten_state_dict
from mani_skill2.utils.visualization.misc import tile_images


class ManiskillWrapper:
    """
    Maniskill環境をDreamerV3で使用するためのラッパー
    """
    metadata = {}

    def __init__(self, env_name, action_repeat=1, size=(64, 64), camera_name="base_camera", seed=0):
        """
        Args:
            env_name: Maniskill環境名 (例: "PickCube-v1", "StackCube-v1")
            action_repeat: 行動の繰り返し回数
            size: 画像サイズ (height, width)
            camera_name: 使用するカメラ名
            seed: 乱数シード
        """
        import mani_skill2.envs
        from mani_skill2.utils.registration import REGISTERED_ENVS

        # 環境の作成
        env_cls = REGISTERED_ENVS[env_name]
        self._env = env_cls(obs_mode="rgbd", control_mode="pd_ee_delta_pose")

        # 設定
        self._action_repeat = action_repeat
        self._size = size
        self._camera_name = camera_name
        self.reward_range = [-np.inf, np.inf]

        # シード設定
        self._env.seed(seed)

        # 初期化
        self._env.reset()

        # 観測空間と行動空間の設定
        self._setup_spaces()

    def _setup_spaces(self):
        """観測空間と行動空間を設定"""
        # 観測空間の設定
        spaces = {}

        # 画像観測（RGB）
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)

        # 状態観測（ロボットの状態など）
        obs = self._env.get_obs()
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key != "image":  # 画像以外の観測
                    if isinstance(value, np.ndarray):
                        shape = value.shape
                    else:
                        shape = (1,)
                    spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        else:
            # 観測が辞書でない場合
            spaces["state"] = gym.spaces.Box(-np.inf, np.inf, obs.shape, dtype=np.float32)

        # DreamerV3で必要な追加フィールド
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
        spaces["is_first"] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(spaces)

        # 行動空間の設定
        self.action_space = self._env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    def step(self, action):
        """環境を1ステップ進める"""
        assert np.isfinite(action).all(), action

        reward = 0
        done = False
        info = {}

        # action_repeat回だけ行動を実行
        for _ in range(self._action_repeat):
            obs, r, d, i = self._env.step(action)
            reward += r
            done = done or d
            info.update(i)
            if done:
                break

        # 観測の処理
        obs = self._process_obs(obs)
        obs["is_terminal"] = np.array([float(done)], dtype=np.float32)
        obs["is_first"] = np.array([0.0], dtype=np.float32)  # step中はFalse

        # 割引率の設定
        if "discount" not in info:
            info["discount"] = np.array(0.0 if done else 1.0, dtype=np.float32)

        return obs, reward, done, info

    def reset(self):
        """環境をリセット"""
        obs = self._env.reset()

        # 観測の処理
        obs = self._process_obs(obs)
        obs["is_terminal"] = np.array([0.0], dtype=np.float32)
        obs["is_first"] = np.array([1.0], dtype=np.float32)  # reset時はTrue

        return obs

    def _process_obs(self, obs):
        """観測をDreamerV3形式に変換"""
        processed_obs = {}

        if isinstance(obs, dict):
            for key, value in obs.items():
                if key == "image":
                    # 画像の前処理
                    processed_obs["image"] = self._process_image(value)
                else:
                    # 状態観測の処理
                    if isinstance(value, np.ndarray):
                        processed_obs[key] = value.astype(np.float32)
                    else:
                        processed_obs[key] = np.array([value], dtype=np.float32)
        else:
            # 観測が辞書でない場合
            processed_obs["state"] = obs.astype(np.float32)
            processed_obs["image"] = self._process_image(self.render())

        return processed_obs

    def _process_image(self, image):
        """画像をDreamerV3用に処理"""
        if len(image.shape) == 4:  # 複数カメラの場合
            # 最初のカメラを使用
            image = image[0]

        # RGBに変換（RGBAの場合はRGBに）
        if image.shape[-1] == 4:
            image = image[..., :3]

        # サイズ調整
        if image.shape[:2] != self._size:
            from PIL import Image
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize(self._size[::-1])  # PILは(width, height)
            image = np.array(image_pil)

        return image.astype(np.uint8)

    def render(self, *args, **kwargs):
        """画像をレンダリング"""
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")

        # Maniskillのレンダリング
        images = self._env.render()

        if isinstance(images, dict):
            # 複数カメラの場合
            if self._camera_name in images:
                return images[self._camera_name]
            else:
                # デフォルトで最初のカメラを使用
                return list(images.values())[0]
        else:
            # 単一画像の場合
            return images

    def close(self):
        """環境を閉じる"""
        self._env.close()

    def seed(self, seed=None):
        """シードを設定"""
        return self._env.seed(seed)


def make_maniskill_env(env_name, action_repeat=1, size=(64, 64), camera_name="base_camera", seed=0):
    """
    Maniskill環境を作成するヘルパー関数

    Args:
        env_name: 環境名
        action_repeat: 行動の繰り返し回数
        size: 画像サイズ
        camera_name: カメラ名
        seed: シード

    Returns:
        ManiskillWrapper: ラップされた環境
    """
    return ManiskillWrapper(env_name, action_repeat, size, camera_name, seed)
