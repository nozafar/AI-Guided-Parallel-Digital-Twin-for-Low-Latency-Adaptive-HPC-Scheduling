from src.schedulers.base import BaseScheduler
from src.digital_twin.job import Job
from src.reinforcement_learning.agent import RLSchedulerAgent
from typing import List
import numpy as np

class RLScheduler(BaseScheduler):
    def __init__(self, env, cluster, simulator):
        super().__init__(env, cluster, simulator)
        
        # We need a trained agent
        # For evaluation, we pass an empty jobs list to the env since we just want the model
        # Normally you'd pass a reference or load it from the path
        self.agent = RLSchedulerAgent({"num_nodes": cluster.num_nodes}, [])
        
    def on_job_arrival(self, job: Job):
        self.env.process(self._schedule_loop())
        
    def on_job_completion(self, job: Job, allocated_nodes: List):
        self.release_resources(allocated_nodes)
        self.env.process(self._schedule_loop())
        
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
        
    def _try_schedule(self) -> bool:
        if not self.simulator.queue:
            return False
            
        state = self._get_state()
        action = self.agent.predict(state)
        
        if action == 0: # Schedule next
            job = self.simulator.queue[0]
            allocated = self.allocate_resources(job)
            if allocated:
                self.env.process(self.simulator.execute_job(job, allocated))
                return True
        elif action == 1: # Delay
            pass # Do nothing
        elif action == 2: # Skip next
            if len(self.simulator.queue) > 1:
                job = self.simulator.queue[1]
                allocated = self.allocate_resources(job)
                if allocated:
                    self.env.process(self.simulator.execute_job(job, allocated))
                    return True
                    
        # Robust Deadlock Prevention
        # If the agent decided to delay, or tried to schedule something that didn't fit,
        # but the cluster is COMPLETELY IDLE, we will freeze if we return False.
        # Force it to act like FCFS as a fallback.
        if not self.simulator.running_jobs:
            job = self.simulator.queue[0]
            allocated = self.allocate_resources(job)
            if allocated:
                self.env.process(self.simulator.execute_job(job, allocated))
                return True
                
        return False
