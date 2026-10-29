import pandas as pd
import numpy as np
import os
import networkx as nx

class SyntheticDataGenerator:
    def __init__(self, num_jobs=100000, seed=42):
        self.num_jobs = num_jobs
        self.seed = seed
        np.random.seed(self.seed)

    def generate_jobs(self, output_path="data/synthetic_hpc_jobs.csv"):
        print(f"Generating {self.num_jobs} synthetic HPC jobs...")
        
        # Arrivals (Poisson process)
        inter_arrival_times = np.random.exponential(scale=60, size=self.num_jobs) # Avg 1 job per minute
        arrival_times = np.cumsum(inter_arrival_times)
        
        # Job types: 0: Normal, 1: CPU-heavy, 2: GPU-heavy, 3: Mixed
        job_types = np.random.choice([0, 1, 2, 3], size=self.num_jobs, p=[0.5, 0.2, 0.2, 0.1])
        
        requested_cpu = np.zeros(self.num_jobs)
        requested_gpu = np.zeros(self.num_jobs)
        requested_memory = np.zeros(self.num_jobs)
        actual_runtime = np.zeros(self.num_jobs)
        
        for i in range(self.num_jobs):
            jt = job_types[i]
            if jt == 0: # Normal
                requested_cpu[i] = np.random.randint(1, 16)
                requested_gpu[i] = 0
                requested_memory[i] = np.random.randint(4, 64)
                actual_runtime[i] = np.random.exponential(scale=3600) # 1 hr avg
            elif jt == 1: # CPU-heavy
                requested_cpu[i] = np.random.randint(16, 128)
                requested_gpu[i] = 0
                requested_memory[i] = np.random.randint(32, 256)
                actual_runtime[i] = np.random.exponential(scale=14400) # 4 hr avg
            elif jt == 2: # GPU-heavy
                requested_cpu[i] = np.random.randint(4, 32)
                requested_gpu[i] = np.random.randint(1, 5)
                requested_memory[i] = np.random.randint(32, 256)
                actual_runtime[i] = np.random.exponential(scale=7200) # 2 hr avg
            else: # Mixed
                requested_cpu[i] = np.random.randint(16, 64)
                requested_gpu[i] = np.random.randint(1, 9)
                requested_memory[i] = np.random.randint(64, 512)
                actual_runtime[i] = np.random.exponential(scale=21600) # 6 hr avg
                
        actual_runtime = np.clip(actual_runtime, 60, 86400 * 7) # Min 1 min, max 7 days
        
        # Estimation errors (users often overestimate)
        overestimation_factor = np.random.uniform(1.0, 3.0, size=self.num_jobs)
        estimated_runtime = actual_runtime * overestimation_factor
        
        priority = np.random.choice([1, 2, 3, 4], size=self.num_jobs, p=[0.7, 0.15, 0.1, 0.05]) # 1 is lowest
        
        # Deadlines
        deadline = arrival_times + estimated_runtime + np.random.exponential(scale=86400, size=self.num_jobs)
        
        # Node count (simple mapping based on CPU request)
        node_count = np.ceil(requested_cpu / 32).astype(int)
        node_count = np.clip(node_count, 1, None)
        
        # Dependency count (most jobs are independent)
        dependency_count = np.random.choice([0, 1, 2, 3], size=self.num_jobs, p=[0.9, 0.05, 0.03, 0.02])
        
        df = pd.DataFrame({
            "job_id": np.arange(self.num_jobs),
            "arrival_time": arrival_times,
            "requested_cpu": requested_cpu.astype(int),
            "requested_gpu": requested_gpu.astype(int),
            "requested_memory": requested_memory.astype(int),
            "runtime": actual_runtime,
            "priority": priority,
            "deadline": deadline,
            "job_type": job_types,
            "node_count": node_count,
            "dependency_count": dependency_count,
            "dependency_graph": [""] * self.num_jobs,
            "estimated_runtime": estimated_runtime,
            "actual_runtime": actual_runtime
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved jobs to {output_path}")
        return df

if __name__ == "__main__":
    generator = SyntheticDataGenerator(num_jobs=100000)
    generator.generate_jobs(output_path="../../data/synthetic_hpc_jobs.csv")
