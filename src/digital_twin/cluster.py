from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Node:
    node_id: int
    total_cpu: int = 32
    total_gpu: int = 4
    total_memory: int = 128
    
    available_cpu: int = 32
    available_gpu: int = 4
    available_memory: int = 128
    
    status: str = "ONLINE" # ONLINE, OFFLINE
    
    def allocate(self, cpu: int, gpu: int, mem: int) -> bool:
        if self.available_cpu >= cpu and self.available_gpu >= gpu and self.available_memory >= mem and self.status == "ONLINE":
            self.available_cpu -= cpu
            self.available_gpu -= gpu
            self.available_memory -= mem
            return True
        return False
        
    def release(self, cpu: int, gpu: int, mem: int):
        self.available_cpu = min(self.total_cpu, self.available_cpu + cpu)
        self.available_gpu = min(self.total_gpu, self.available_gpu + gpu)
        self.available_memory = min(self.total_memory, self.available_memory + mem)

class Cluster:
    def __init__(self, num_nodes: int = 64, cpu_per_node: int = 32, gpu_per_node: int = 4, mem_per_node: int = 128):
        self.nodes = [Node(i, cpu_per_node, gpu_per_node, mem_per_node, cpu_per_node, gpu_per_node, mem_per_node) for i in range(num_nodes)]
        self.num_nodes = num_nodes
        
    def get_total_resources(self):
        total_cpu = sum(n.total_cpu for n in self.nodes)
        total_gpu = sum(n.total_gpu for n in self.nodes)
        total_mem = sum(n.total_memory for n in self.nodes)
        return total_cpu, total_gpu, total_mem
        
    def get_available_resources(self):
        avail_cpu = sum(n.available_cpu for n in self.nodes if n.status == "ONLINE")
        avail_gpu = sum(n.available_gpu for n in self.nodes if n.status == "ONLINE")
        avail_mem = sum(n.available_memory for n in self.nodes if n.status == "ONLINE")
        return avail_cpu, avail_gpu, avail_mem
        
    def get_utilization(self):
        total_cpu, total_gpu, total_mem = self.get_total_resources()
        avail_cpu, avail_gpu, avail_mem = self.get_available_resources()
        
        return {
            "cpu": (total_cpu - avail_cpu) / total_cpu if total_cpu > 0 else 0,
            "gpu": (total_gpu - avail_gpu) / total_gpu if total_gpu > 0 else 0,
            "memory": (total_mem - avail_mem) / total_mem if total_mem > 0 else 0
        }
