import simpy
import pandas as pd
import numpy as np
from typing import List, Dict, Type
from .job import Job
from .cluster import Cluster

class Simulator:
    def __init__(self, cluster: Cluster, scheduler_class: Type, jobs: List[Job], config: Dict = None):
        self.env = simpy.Environment()
        self.cluster = cluster
        self.jobs = sorted(jobs, key=lambda j: j.arrival_time)
        self.config = config or {}
        
        self.scheduler = scheduler_class(self.env, self.cluster, self)
        
        self.queue: List[Job] = []
        self.running_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        
        # Metrics tracking
        self.metrics = []
        
        # Start processes
        self.env.process(self.job_generator())
        self.env.process(self.metrics_collector())
        
    def job_generator(self):
        for job in self.jobs:
            # Wait until arrival time
            if job.arrival_time > self.env.now:
                yield self.env.timeout(job.arrival_time - self.env.now)
                
            self.queue.append(job)
            self.scheduler.on_job_arrival(job)
            
    def execute_job(self, job: Job, allocated_nodes: List[int]):
        job.start_time = self.env.now
        job.status = "RUNNING"
        self.running_jobs.append(job)
        self.queue.remove(job)
        
        # Resource allocation happens in the scheduler, we just wait for runtime
        yield self.env.timeout(job.actual_runtime)
        
        job.end_time = self.env.now
        job.status = "COMPLETED"
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)
        
        self.scheduler.on_job_completion(job, allocated_nodes)
        
    def metrics_collector(self, interval=60):
        while len(self.completed_jobs) < len(self.jobs):
            # Deadlock detection: if queue has jobs, nothing is running, and no jobs will arrive in the future
            if self.queue and not self.running_jobs:
                future_arrivals = any(j.arrival_time > self.env.now for j in self.jobs if j not in self.queue and j not in self.completed_jobs and j not in self.running_jobs)
                if not future_arrivals:
                    print(f"Deadlock detected at time {self.env.now}. Aborting simulation.")
                    break
                    
            util = self.cluster.get_utilization()
            self.metrics.append({
                "time": self.env.now,
                "queue_length": len(self.queue),
                "running_jobs": len(self.running_jobs),
                "cpu_util": util["cpu"],
                "gpu_util": util["gpu"],
                "memory_util": util["memory"]
            })
            yield self.env.timeout(interval)
            
    def run(self, until=None):
        self.env.run(until=until)
        return self.get_results()
        
    def get_results(self):
        return {
            "completed_jobs": self.completed_jobs,
            "metrics": pd.DataFrame(self.metrics)
        }
