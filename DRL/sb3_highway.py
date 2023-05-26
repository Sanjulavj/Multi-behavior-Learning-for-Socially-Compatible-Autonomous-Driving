# Environment
import gym

# Agent
from stable_baselines3 import DQN

import sys
from tqdm.notebook import trange

sys.path.insert(0, '/Users/sanjulavj/Desktop/PRj/Test Folder4 Highway/highway-env-1.3/scripts/')
from utils import record_videos, show_videos
from stable_baselines3.common.callbacks import CheckpointCallback
import highway_env.envs.highway_env


"""Training

"""

print("start...")

model = DQN('MlpPolicy', "highway-v0",
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=100,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="highway_dqn/") 

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq= 1000,
  save_path="./save/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
) 


#model.learn(int(25e3), callback = checkpoint_callback)

"""Testing

Visualize a few episodes
"""
print("evaluate...")
  
env = gym.make("highway-v0")
model_path = "/Users/sanjulavj/Desktop/PRj/Multi-behavior-Learning-for-Socially-Compatible-Autonomous-Driving/DRL/save/rl_model_21000_steps"
model_eval = DQN.load(model_path, env=env)

env = record_videos(env)
for episode in trange(38, desc="Test episodes"):
    obs, done = env.reset(), False
    while not done:
        action, _ = model_eval.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
env.close()
show_videos()  