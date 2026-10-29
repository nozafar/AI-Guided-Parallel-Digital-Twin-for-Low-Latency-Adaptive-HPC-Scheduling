from src.schedulers.base import BaseScheduler
from src.digital_twin.job import Job
from typing import List

class FCFSScheduler(BaseScheduler):
    def on_job_arrival(self, job: Job):
        self.env.process(self._schedule_loop())
        
    def on_job_completion(self, job: Job, allocated_nodes: List):
        self.release_resources(allocated_nodes)
        self.env.process(self._schedule_loop())
        
    def _try_schedule(self) -> bool:
        if not self.simulator.queue:
            return False
            
        # First Come First Serve (queue is already sorted by arrival time)
        job = self.simulator.queue[0]
        allocated = self.allocate_resources(job)
        
        if allocated:
            self.env.process(self.simulator.execute_job(job, allocated))
            return True
            
        return False

class SJFScheduler(BaseScheduler):
    def on_job_arrival(self, job: Job):
        self.env.process(self._schedule_loop())
        
    def on_job_completion(self, job: Job, allocated_nodes: List):
        self.release_resources(allocated_nodes)
        self.env.process(self._schedule_loop())
        
    def _try_schedule(self) -> bool:
        if not self.simulator.queue:
            return False
            
        # Shortest Job First (using estimated runtime)
        sorted_queue = sorted(self.simulator.queue, key=lambda j: j.estimated_runtime)
        
        job = sorted_queue[0]
        allocated = self.allocate_resources(job)
        
        if allocated:
            self.env.process(self.simulator.execute_job(job, allocated))
            return True
            
        return False
        
class PriorityScheduler(BaseScheduler):
    def on_job_arrival(self, job: Job):
        self.env.process(self._schedule_loop())
        
    def on_job_completion(self, job: Job, allocated_nodes: List):
        self.release_resources(allocated_nodes)
        self.env.process(self._schedule_loop())
        
    def _try_schedule(self) -> bool:
        if not self.simulator.queue:
            return False
            
        # Priority (higher number is higher priority), break ties by arrival time
        sorted_queue = sorted(self.simulator.queue, key=lambda j: (-j.priority, j.arrival_time))
        
        job = sorted_queue[0]
        allocated = self.allocate_resources(job)
        
        if allocated:
            self.env.process(self.simulator.execute_job(job, allocated))
            return True
            
        return False
