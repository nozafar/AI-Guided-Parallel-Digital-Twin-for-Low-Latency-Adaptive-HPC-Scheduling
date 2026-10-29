import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.digital_twin.cluster import Cluster
from src.digital_twin.job import Job
from src.digital_twin.simulator import Simulator
from typing import List, Dict

from src.schedulers.base import BaseScheduler
import copy

class EnvScheduler(BaseScheduler):
    def on_job_arrival(self, job): pass
    def on_job_completion(self, job, alloc): pass
    def _try_schedule(self): return False

class HPCSchedulingEnv(gym.Env):
    """Custom Environment for HPC Scheduling using Gym."""
    metadata = {'render.modes': ['human']}

    def __init__(self, cluster_config: Dict, jobs: List[Job]):
        super(HPCSchedulingEnv, self).__init__()
        
        self.cluster_config = cluster_config
        self.jobs = jobs
        
        # Actions: 
        # 0: schedule next job
        # 1: delay next job
        # 2: skip next job and schedule the one after
        self.action_space = spaces.Discrete(3)
        
        # State representation:
        # [queue_length, avail_cpu, avail_gpu, avail_mem, 
        #  job1_cpu, job1_gpu, job1_mem, job1_est_runtime, ... up to job5]
        self.observation_space = spaces.Box(low=0, high=100000, shape=(24,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cluster = Cluster(**self.cluster_config)
        
        # Create a proxy scheduler that delegates to this env
        self.simulator = Simulator(self.cluster, EnvScheduler, copy.deepcopy(self.jobs))
        self.env_sim = self.simulator.env
        
        # Run until first job arrives
        self._step_until_job()
        
        self.state = self._get_state()
        return self.state, {}
        
    def _step_until_job(self):
        while self.env_sim.peek() < float('inf') and not self.simulator.queue:
            self.env_sim.step()
            
    def _get_state(self):
        q_len = len(self.simulator.queue)
        avail_cpu, avail_gpu, avail_mem = self.cluster.get_available_resources()
        
        state = [float(q_len), float(avail_cpu), float(avail_gpu), float(avail_mem)]
        
        for i in range(5):
            if i < q_len:
                j = self.simulator.queue[i]
                state.extend([float(j.requested_cpu), float(j.requested_gpu), float(j.requested_memory), float(j.estimated_runtime)])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])
                
        return np.array(state, dtype=np.float32)
        
    def step(self, action):
        reward = 0
        done = False
        
        if not self.simulator.queue:
            self._step_until_job()
            
        alloc = False
        if self.simulator.queue:
            if action == 0: # Schedule next
                job = self.simulator.queue[0]
                alloc = self.simulator.scheduler.allocate_resources(job)
                if alloc:
                    self.env_sim.process(self.simulator.execute_job(job, alloc))
                    reward += 5 # small bonus for successful action
                else:
                    reward -= 5 # penalty for invalid action
            elif action == 1: # Delay
                reward -= 1 
            elif action == 2: # Skip and schedule next
                if len(self.simulator.queue) > 1:
                    job = self.simulator.queue[1]
                    alloc = self.simulator.scheduler.allocate_resources(job)
                    if alloc:
                        self.env_sim.process(self.simulator.execute_job(job, alloc))
                        reward += 5
                    else:
                        reward -= 5
                else:
                    reward -= 5
                    
        completed_before = len(self.simulator.completed_jobs)
        
        # We must process all events at the CURRENT timestamp immediately 
        # so that execute_job runs and removes the scheduled job from the queue!
        current_time = self.env_sim.now
        while self.env_sim.peek() == current_time:
            self.env_sim.step()
        
        # Advance simulation to the NEXT event ONLY if delay, or failed to allocate, or queue is empty
        if action == 1 or not alloc or not self.simulator.queue:
            target_time = current_time + 1.0 # Or we can just step until time changes
            while self.env_sim.peek() < target_time:
                self.env_sim.step()
                
        completed_after = len(self.simulator.completed_jobs)
        reward += (completed_after - completed_before) * 100 # Positive reward for finishing a job!
            
        if self.env_sim.peek() == float('inf'):
            if not self.simulator.queue and not self.simulator.running_jobs:
                done = True
                reward += 1000 # Bonus for finishing all jobs
            elif self.simulator.queue and not self.simulator.running_jobs:
                # Deadlock detected! End the episode and penalize heavily
                done = True
                reward -= 100000 # Massive penalty so it doesn't choose suicide
            
        # Reward based on waiting time and makespan will be computed
        for job in self.simulator.queue:
            reward -= 0.1 # Penalty for jobs sitting in queue
            
        self.state = self._get_state()
        return self.state, reward, done, False, {}
