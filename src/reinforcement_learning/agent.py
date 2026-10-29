from stable_baselines3 import PPO
from src.reinforcement_learning.env import HPCSchedulingEnv
import os

class RLSchedulerAgent:
    def __init__(self, cluster_config, jobs, model_path="models/ppo_scheduler.zip"):
        self.env = HPCSchedulingEnv(cluster_config, jobs)
        self.model_path = model_path
        
        if os.path.exists(self.model_path):
            print("Loading existing PPO model...")
            self.model = PPO.load(self.model_path, env=self.env, device="cuda")
        else:
            print("Creating new PPO model...")
            self.model = PPO("MlpPolicy", self.env, verbose=1, device="cuda")
            
    def train(self, total_timesteps=10000):
        print(f"Training PPO for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def predict(self, state):
        action, _states = self.model.predict(state, deterministic=True)
        return action
