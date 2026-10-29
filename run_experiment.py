import argparse
import pandas as pd
import yaml
from src.data.synthetic_generator import SyntheticDataGenerator
from src.prediction.model import WorkloadPredictor
from src.reinforcement_learning.agent import RLSchedulerAgent
from src.digital_twin.job import Job
from src.digital_twin.cluster import Cluster
from src.digital_twin.simulator import Simulator
from src.schedulers.baselines import FCFSScheduler, SJFScheduler, PriorityScheduler
from src.schedulers.rl_scheduler import RLScheduler
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()
    
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_jobs(data_path, num_jobs=1000):
    df = pd.read_csv(data_path, nrows=num_jobs)
    # df = df.head(num_jobs) # For fast simulation
    
    jobs = []
    for _, row in df.iterrows():
        job = Job(
            job_id=int(row['job_id']),
            arrival_time=float(row['arrival_time']),
            requested_cpu=int(row['requested_cpu']),
            requested_gpu=int(row['requested_gpu']),
            requested_memory=int(row['requested_memory']),
            runtime=float(row['runtime']),
            priority=int(row['priority']),
            deadline=float(row['deadline']),
            job_type=int(row['job_type']),
            node_count=int(row['node_count']),
            estimated_runtime=float(row['estimated_runtime']),
            actual_runtime=float(row['actual_runtime'])
        )
        jobs.append(job)
    return jobs

def main():
    args = parse_args()
    config = load_config(args.config)
    
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Generate Data
    if not os.path.exists("data/synthetic_hpc_jobs.csv"):
        generator = SyntheticDataGenerator(num_jobs=5000, seed=config['random_seed'])
        generator.generate_jobs("data/synthetic_hpc_jobs.csv")
        
    # 2. Train Predictor
    predictor = WorkloadPredictor()
    if not os.path.exists("models/xgboost_predictor.joblib"):
        predictor.train("data/synthetic_hpc_jobs.csv")
        
    jobs = load_jobs("data/synthetic_hpc_jobs.csv", num_jobs=config.get('sim_jobs', 1000))
    
    # 3. Train RL Agent
    cluster_config = config['cluster']
    agent = RLSchedulerAgent(cluster_config, jobs)
    if not os.path.exists("models/ppo_scheduler.zip"):
        agent.train(total_timesteps=config.get('rl_timesteps', 5000))
        
    # 4. Run Evaluation
    schedulers = {
        "FCFS": FCFSScheduler,
        "SJF": SJFScheduler,
        "Priority": PriorityScheduler,
        "RL": RLScheduler
    }
    
    results = {}
    
    for name, scheduler_class in schedulers.items():
        print(f"Running simulation with {name} scheduler...")
        cluster = Cluster(**cluster_config)
        # Deep copy jobs
        sim_jobs = load_jobs("data/synthetic_hpc_jobs.csv", num_jobs=config.get('sim_jobs', 1000))
        
        sim = Simulator(cluster, scheduler_class, sim_jobs)
        start_time = time.time()
        sim_results = sim.run()
        print(f"{name} simulation completed in {time.time() - start_time:.2f}s")
        
        comp_jobs = sim_results['completed_jobs']
        avg_wait = sum(j.queue_time for j in comp_jobs) / len(comp_jobs) if comp_jobs else 0
        makespan = max(j.end_time for j in comp_jobs) if comp_jobs else 0
        
        results[name] = {
            "avg_waiting_time": avg_wait,
            "makespan": makespan
        }
        print(f"{name} Results: Avg Wait: {avg_wait:.2f}s, Makespan: {makespan:.2f}s")
        
    results_df = pd.DataFrame(results).T
    results_df.to_csv("results/tables/simulation_results.csv")
    print("Experiments completed. Results saved to results/tables/simulation_results.csv")

if __name__ == "__main__":
    main()
