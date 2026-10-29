from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Job:
    job_id: int
    arrival_time: float
    requested_cpu: int
    requested_gpu: int
    requested_memory: int
    runtime: float
    priority: int
    deadline: float
    job_type: int
    node_count: int
    estimated_runtime: float
    actual_runtime: float
    
    # State tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "PENDING" # PENDING, RUNNING, COMPLETED, FAILED
    
    # Dependencies (IDs)
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    
    @property
    def queue_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time
    
    @property
    def turnaround_time(self) -> float:
        if self.end_time is None:
            return 0.0
        return self.end_time - self.arrival_time
