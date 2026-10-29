from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.digital_twin.job import Job
from src.digital_twin.cluster import Cluster

class BaseScheduler(ABC):
    def __init__(self, env, cluster: Cluster, simulator):
        self.env = env
        self.cluster = cluster
        self.simulator = simulator
        self.is_scheduling = False
        
    @abstractmethod
    def on_job_arrival(self, job: Job):
        pass
        
    @abstractmethod
    def on_job_completion(self, job: Job, allocated_nodes: List[int]):
        pass
        
    def _schedule_loop(self):
        if self.is_scheduling:
            return
        self.is_scheduling = True
        
        while True:
            # Subclasses should implement the scheduling logic here
            jobs_scheduled = self._try_schedule()
            if not jobs_scheduled:
                break
            # Small delay to allow state updates
            yield self.env.timeout(0.01)
            
        self.is_scheduling = False
        
    @abstractmethod
    def _try_schedule(self) -> bool:
        """Attempt to schedule jobs. Return True if at least one job was scheduled."""
        pass
        
    def allocate_resources(self, job: Job) -> List[int]:
        # Simple greedy allocation
        allocated_nodes = []
        cpu_needed = job.requested_cpu
        gpu_needed = job.requested_gpu
        mem_needed = job.requested_memory
        
        # Check if enough total resources are available
        avail_cpu, avail_gpu, avail_mem = self.cluster.get_available_resources()
        if avail_cpu < cpu_needed or avail_gpu < gpu_needed or avail_mem < mem_needed:
            return []
            
        # Greedily allocate from nodes
        for node in self.cluster.nodes:
            if node.status != "ONLINE":
                continue
                
            alloc_cpu = min(node.available_cpu, cpu_needed)
            alloc_gpu = min(node.available_gpu, gpu_needed)
            alloc_mem = min(node.available_memory, mem_needed)
            
            if alloc_cpu > 0 or alloc_gpu > 0 or alloc_mem > 0:
                if node.allocate(alloc_cpu, alloc_gpu, alloc_mem):
                    allocated_nodes.append((node.node_id, alloc_cpu, alloc_gpu, alloc_mem))
                    cpu_needed -= alloc_cpu
                    gpu_needed -= alloc_gpu
                    mem_needed -= alloc_mem
                    
            if cpu_needed == 0 and gpu_needed == 0 and mem_needed == 0:
                break
                
        if cpu_needed == 0 and gpu_needed == 0 and mem_needed == 0:
            return allocated_nodes
            
        # Rollback if we couldn't fully allocate
        self.release_resources(allocated_nodes)
        return []
        
    def release_resources(self, allocated_nodes: List[Any]):
        for node_id, cpu, gpu, mem in allocated_nodes:
            self.cluster.nodes[node_id].release(cpu, gpu, mem)
