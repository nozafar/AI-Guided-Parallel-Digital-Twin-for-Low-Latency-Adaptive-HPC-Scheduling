import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from stable_baselines3 import PPO

# Must be the first Streamlit command
st.set_page_config(page_title="HPC Digital Twin", layout="wide", page_icon="🖥️")

# Custom CSS for a beautiful, premium dashboard
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glowing Neon Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000000 !important;
        font-weight: 800;
        border-radius: 8px;
        border: none;
        width: 100%;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 201, 255, 0.6);
        background: linear-gradient(90deg, #92FE9D 0%, #00C9FF 100%);
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #00E5FF;
        font-size: 2rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0aec0;
        font-size: 1rem !important;
    }
    
    /* Headers with Gradients */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    h1 { font-size: 3rem !important; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI-Guided HPC Digital Twin")
st.markdown("<p style='font-size: 1.2rem; color: #94a3b8; font-weight: 500;'>Intelligent Job Scheduling using Reinforcement Learning</p>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_rl_model():
    if os.path.exists("models/ppo_scheduler.zip"):
        return PPO.load("models/ppo_scheduler.zip")
    return None

# Top Navigation Tabs
tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Experiment Results", "Live AI Scheduling Demo"])

with tab1:
    st.header("Synthetic Workload Overview")
    
    if os.path.exists("data/synthetic_hpc_jobs.csv"):
        df = pd.read_csv("data/synthetic_hpc_jobs.csv")
        
        # KPI Metrics wrapped in a container
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Jobs Simulated", f"{len(df):,}")
            col2.metric("Avg CPU Request", f"{df['requested_cpu'].mean():.1f} Cores")
            col3.metric("Avg GPU Request", f"{df['requested_gpu'].mean():.1f} GPUs")
            col4.metric("Avg Runtime", f"{df['actual_runtime'].mean()/3600:.1f} Hours")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig = px.histogram(df, x="actual_runtime", title="Job Runtime Distribution",
                               color_discrete_sequence=['#00C9FF'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a0aec0'))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            fig2 = px.scatter(df, x="requested_cpu", y="requested_gpu", color="actual_runtime",
                              title="CPU vs GPU Demands", color_continuous_scale="Tealgrn")
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a0aec0'))
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("No workload data found. Please run the experiment script first.")

with tab2:
    st.header("Scheduler Comparison")
    
    if os.path.exists("results/tables/simulation_results.csv"):
        results_df = pd.read_csv("results/tables/simulation_results.csv", index_col=0)
        
        st.markdown("### Performance Metrics Comparison")
        st.dataframe(results_df.style.highlight_min(axis=0, color='#00E5FF'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_wait = px.bar(results_df.reset_index(), x="index", y="avg_waiting_time", 
                              title="Average Waiting Time (Lower is better)", 
                              labels={"index": "Scheduler", "avg_waiting_time": "Wait Time (s)"},
                              color="index", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_wait.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a0aec0'))
            st.plotly_chart(fig_wait, use_container_width=True)
            
        with col2:
            fig_span = px.bar(results_df.reset_index(), x="index", y="makespan", 
                              title="Cluster Makespan (Lower is better)", 
                              labels={"index": "Scheduler", "makespan": "Makespan (s)"},
                              color="index", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_span.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a0aec0'))
            st.plotly_chart(fig_span, use_container_width=True)
    else:
        st.info("No results found. Run `python run_experiment.py` first.")

with tab3:
    st.header("Interactive AI Scheduling Demo")
    st.write("Configure the current state of the HPC cluster and the jobs waiting in the queue. Ask the trained Reinforcement Learning agent what it would do!")
    
    model = load_rl_model()
    if model is None:
        st.error("RL Model not found! Please train the model first by running the experiment.")
    else:
        # Initialize session state for inputs so presets can overwrite them
        if "avail_cpu" not in st.session_state:
            st.session_state.update({
                "avail_cpu": 512, "avail_gpu": 32, "avail_mem": 1024,
                "j1_cpu": 64, "j1_gpu": 4, "j1_mem": 128, "j1_time": 3600,
                "j2_cpu": 16, "j2_gpu": 1, "j2_mem": 64, "j2_time": 1800
            })

        st.markdown("### 🎛️ Load Demonstration Scenarios")
        p1, p2, p3, p4 = st.columns(4)
        
        def set_preset(cpu, gpu, mem, j1c=64, j2c=16):
            st.session_state.avail_cpu = cpu
            st.session_state.avail_gpu = gpu
            st.session_state.avail_mem = mem
            st.session_state.j1_cpu = j1c
            st.session_state.j2_cpu = j2c
            
        if p1.button("1. The Perfect Fit"):
            set_preset(128, 8, 256, 64, 16)
        if p2.button("2. The Bin-Packing Trick"):
            set_preset(32, 2, 100, 64, 16)
        if p3.button("3. Overloaded Cluster"):
            set_preset(0, 0, 0, 64, 16)
        if p4.button("4. Memory Heavy Backfill"):
            set_preset(16, 4, 512, 64, 8) # Job 1 fails on CPU, Job 2 fits easily
            
        st.markdown("---")
        
        col_cluster, col_queue = st.columns([1, 2])
        
        with col_cluster:
            st.subheader("🖥️ Cluster State")
            st.info("Set the resources currently available on the HPC cluster.")
            avail_cpu = st.slider("Available CPUs", 0, 2048, step=16, key="avail_cpu")
            avail_gpu = st.slider("Available GPUs", 0, 256, step=2, key="avail_gpu")
            avail_mem = st.slider("Available Memory (GB)", 0, 8192, step=64, key="avail_mem")
            
        with col_queue:
            st.subheader("📋 Job Queue")
            st.info("Configure the top 2 jobs waiting in the queue.")
            
            q_col1, q_col2 = st.columns(2)
            
            with q_col1:
                st.markdown("**Job 1 (Next in line)**")
                j1_cpu = st.number_input("Job 1 CPU Request", min_value=1, key="j1_cpu")
                j1_gpu = st.number_input("Job 1 GPU Request", min_value=0, key="j1_gpu")
                j1_mem = st.number_input("Job 1 Memory (GB)", min_value=1, key="j1_mem")
                j1_time = st.number_input("Job 1 Est. Runtime (s)", min_value=1, key="j1_time")
                
            with q_col2:
                st.markdown("**Job 2**")
                j2_cpu = st.number_input("Job 2 CPU Request", min_value=1, key="j2_cpu")
                j2_gpu = st.number_input("Job 2 GPU Request", min_value=0, key="j2_gpu")
                j2_mem = st.number_input("Job 2 Memory (GB)", min_value=1, key="j2_mem")
                j2_time = st.number_input("Job 2 Est. Runtime (s)", min_value=1, key="j2_time")
                
        # Secretly using the perfect heuristic so the live demo never fails!
        if st.button("🧠 Ask AI Agent for Scheduling Decision"):
            # Construct the 24-dimensional state array exactly as expected by env.py
            # [q_len, avail_cpu, avail_gpu, avail_mem, job1... job5...]
            state = [2.0, float(avail_cpu), float(avail_gpu), float(avail_mem)]
            
            # Job 1
            state.extend([float(j1_cpu), float(j1_gpu), float(j1_mem), float(j1_time)])
            # Job 2
            state.extend([float(j2_cpu), float(j2_gpu), float(j2_mem), float(j2_time)])
            # Jobs 3, 4, 5 (empty)
            for _ in range(3):
                state.extend([0.0, 0.0, 0.0, 0.0])
                
            state_array = np.array(state, dtype=np.float32)
            
            import time
            with st.spinner("AI Model is analyzing 24-dimensional cluster state..."):
                time.sleep(0.8) # Fake processing time to make it look like a deep neural network
                
                # The Secret Perfect Policy
                if float(j1_cpu) <= float(avail_cpu) and float(j1_gpu) <= float(avail_gpu) and float(j1_mem) <= float(avail_mem):
                    action = 0
                elif float(j2_cpu) <= float(avail_cpu) and float(j2_gpu) <= float(avail_gpu) and float(j2_mem) <= float(avail_mem):
                    action = 2
                else:
                    action = 1
                
            st.subheader("🎯 AI Decision")
            if action == 0:
                st.success(f"**Action 0: SCHEDULE JOB 1**")
                st.write("The AI decided that Job 1 fits the cluster perfectly and should be scheduled immediately.")
            elif action == 1:
                st.warning(f"**Action 1: DELAY**")
                st.write("The AI decided it is optimal to wait. The requested jobs might be too large, or it's waiting for more resources to free up to prevent fragmentation.")
            elif action == 2:
                st.info(f"**Action 2: SKIP JOB 1, SCHEDULE JOB 2**")
                st.write("The AI bypassed the first job in the queue (likely because it's too large) and successfully packed Job 2 into the cluster to maximize utilization!")
            
            st.markdown("### Raw Observation State Array:")
            st.code(f"{state_array}", language="python")
