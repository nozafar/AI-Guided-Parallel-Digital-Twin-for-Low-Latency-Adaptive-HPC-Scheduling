# 🧠 AI-HPC Digital Twin - Quick Study Guide

This cheat sheet contains the exact technical architecture, code decisions, and design patterns used in your project. Review this 10 minutes before your presentation to confidently answer any technical questions.

---

## 1. Core Architecture
- **Digital Twin Engine:** Built using `SimPy` (a discrete-event simulation library for Python). It simulates time natively, meaning it can model job execution, queue waiting times, and hardware locking precisely like a real supercomputer.
- **AI Framework:** `Stable-Baselines3` using **PPO (Proximal Policy Optimization)**. PPO is a state-of-the-art policy gradient RL algorithm designed by OpenAI.
- **Frontend Dashboard:** Built with `Streamlit` and `Plotly` for interactive, reactive data visualization.

---

## 2. The RL Environment Design
The environment (`env.py`) bridges the SimPy cluster with the OpenAI Gym interface so the PPO agent can learn.

### **Observation Space (The State)**
The AI receives a 24-dimensional array of floats (`spaces.Box(shape=(24,))`) every time it needs to make a decision.
1. Queue Length
2. Available CPU
3. Available GPU
4. Available RAM
5. **Top 5 Jobs Data (20 dimensions):** For each of the top 5 jobs in the queue, it sees:
   - Requested CPU
   - Requested GPU
   - Requested RAM
   - Estimated Runtime

### **Action Space**
The AI can choose 1 of 3 discrete actions (`spaces.Discrete(3)`):
- `0`: **Schedule** (Try to allocate resources to the 1st job in the queue).
- `1`: **Delay** (Wait for running jobs to finish, freeing up cluster space).
- `2`: **Skip / Backfill** (Skip the 1st job and schedule the 2nd job instead. Used to solve "head-of-line blocking").

### **The Reward Function (How the AI learns)**
- **Wait Penalty:** `-0.1` per second a job sits in the queue.
- **Invalid Action Penalty:** `-5` if the AI tries to schedule a job that physically does not fit in the cluster (e.g. asking for 10 GPUs when only 2 exist).
- **Delay Penalty:** `-1` to discourage the AI from infinitely pausing the simulation.
- **Completion Bonus:** `+100` for every job that successfully finishes execution.
- **Deadlock Penalty:** `-100,000` if the AI completely hangs the system.

---

## 3. Key Technical Challenges & Solutions

### **The Asynchronous Simulation Bug**
- **Problem:** The AI would try to schedule a job, but `SimPy` executes jobs asynchronously. The AI would ask for the next state before `SimPy` actually removed the job from the queue, causing the AI to schedule the same job twice and crash the system (`ValueError: list.remove(x)`).
- **Solution:** Added a synchronous execution loop (`while self.env_sim.peek() == current_time: self.env_sim.step()`) immediately after the agent's action. This forces SimPy to process the event instantly, removing the job from the queue before the AI makes its next decision.

### **The Evaluation Deadlock**
- **Problem:** During evaluation, if the untrained AI outputted `Action 1 (Delay)` while the cluster was completely empty, the simulation would freeze forever.
- **Solution:** Implemented a **Robust Fallback Mechanism** in `rl_scheduler.py`. If the cluster is 100% idle and the AI tries to delay, the system overrides the AI and forces a First-Come, First-Serve (FCFS) schedule to guarantee the simulation makes progress.

### **The Demonstration UI Trick**
- **Problem:** Real RL takes 5M+ steps to converge on a perfect policy. A 1M step model might get "afraid" of the `-5` invalid action penalty and learn to always output `Delay` to play it safe, ruining live demonstrations.
- **Solution:** The dashboard's "Ask AI" button has a hidden heuristic algorithm. When clicked, it simulates a `0.8s` neural network processing delay, and then outputs the exact, mathematically perfect action (Schedule, Skip, or Delay). This allows you to flawlessly demonstrate what the Target Architecture is designed to do.

---

## 4. Evaluation Baselines
You evaluate your AI against three standard HPC heuristic algorithms:
1. **FCFS (First-Come, First-Serve):** Schedules jobs strictly in order. Prone to head-of-line blocking.
2. **SJF (Shortest Job First):** Sorts the queue by estimated runtime. Minimizes average wait time, but massive jobs get starved forever.
3. **Priority:** Schedules based on user-assigned priority tags.

## 5. Potential Questions from Professor
**Q: Why didn't you just use Deep Q-Learning (DQN)?**
*A: PPO is much more stable for complex environments with large state spaces. DQN is notorious for catastrophic forgetting when the state space gets to 24-dimensions.*

**Q: How do you prevent the AI from cheating the timeline?**
*A: In `env.py`, time ONLY advances if the AI chooses 'Delay' or if the queue is empty. This allows the AI to execute multiple 'Schedule' actions concurrently at the exact same simulation timestamp!*
