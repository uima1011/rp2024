from stable_baselines3 import PPO
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
from pprint import pprint

MODEL_PATH = 'data/models/PPO/10000.zip'
ENV = svpEnv()

def test_model(model_path, env):
    model = PPO.load(model_path)
    obs, _info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        pprint(info)
        if terminated or truncated:
            obs = env.reset()

if __name__ == "__main__":
    test_model(MODEL_PATH, ENV)