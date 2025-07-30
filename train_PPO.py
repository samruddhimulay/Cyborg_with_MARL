from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Environment import DefenderCyborgEnv
import time
from torch.utils.tensorboard import SummaryWriter
import os
from Callbacks import *

# Create and wrap your environment
env = make_vec_env(DefenderCyborgEnv, n_envs=4)  # If you have multiple CPU cores, increase n_envs
logdir = "logs/train_PPO_" + time.strftime("%Y%m%d_%H%M%S")

# Instantiate the PPO model (MlpPolicy is suitable for vector observations)
model = PPO("MlpPolicy", env, verbose=1, seed=42,tensorboard_log=logdir)  # Set verbose to see progress


info_callback = InfoLoggingCallback(verbose=1)

# Train for 10,000 timesteps (adjust to fit your needs)
model.learn(total_timesteps=3_000_000,callback = info_callback)

# Save the model
model.save("ppo_cyborg_agent")
