import streamlit as st
import numpy as np

def erlang_a_stationary_distribution(λ, μ, γ, s, max_calls):
    """Calculate stationary distribution for Erlang-A call center model"""
    Q = np.zeros((max_calls + 1, max_calls + 1))
    
    for i in range(max_calls + 1):
        if i < max_calls:
            Q[i, i+1] = λ  # Arrivals
        
        if i > 0:
            if i <= s:
                Q[i, i-1] = i * μ  # Service
            else:
                Q[i, i-1] = s * μ + (i - s) * γ  # Service + abandonment
        
        Q[i, i] = -np.sum(Q[i, :])
    
    A = Q.T
    A[-1, :] = 1  # Normalization condition
    b = np.zeros(max_calls + 1)
    b[-1] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return {
        'delay_probability': sum(pi[s+1:]),
        'agent_utilization': sum(min(i, s)/s * pi[i] for i in range(1, max_calls+1))
    }

def find_min_agents(λ, μ, γ, max_delay_prob, max_calls):
    """Find minimum agents needed by incremental search"""
    s = 1
    while True:
        try:
            metrics = erlang_a_stationary_distribution(λ, μ, γ, s, max_calls)
            if metrics['delay_probability'] <= max_delay_prob:
                return s
            s += 1
            if s > max_calls:  # Prevent infinite loop
                return max_calls
        except:
            return max_calls  # Fallback if calculation fails

def main():
    st.title("Erlang-A Call Center ")
    
    st.header("System Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        λ = st.number_input("Arrival rate (λ)", 
                           min_value=0.1, value=10.0, step=0.1)
        μ = st.number_input("Service rate (μ)", 
                           min_value=0.1, value=0.5, step=0.1)
        γ = st.number_input("Abandonment rate (γ)", 
                           min_value=0.01, value=0.1, step=0.01)
    
    with col2:
        max_calls = st.number_input("System capacity (max calls)", 
                                  min_value=2, value=50, step=1)
        max_delay_prob_pct = st.number_input("Max delay probability (%)", 
                                            min_value=1.0, max_value=100.0, 
                                            value=20.0, step=0.1,
                                            format="%.1f")
        max_delay_prob = max_delay_prob_pct / 100.0  # Convert to decimal for calculations
    
    if st.button("Calculate Required Staffing"):
        try:
            min_agents = find_min_agents(λ, μ, γ, max_delay_prob, max_calls)
            
            st.header("Results")
            st.success(f"## Minimum agents required: {min_agents}")
            
            # Calculate metrics for found agent count
            metrics = erlang_a_stationary_distribution(λ, μ, γ, min_agents, max_calls)
            
            st.subheader("Performance at Minimum Staffing")
            col1, col2 = st.columns(2)
            col1.metric("Actual delay probability", 
                       f"{metrics['delay_probability']*100:.1f}%",
                       help="Should be ≤ target delay probability")
            col2.metric("Agent utilization", 
                       f"{metrics['agent_utilization']*100:.1f}%")
            
            # Show what happens with one less agent
            if min_agents > 1:
                prev_metrics = erlang_a_stationary_distribution(λ, μ, γ, min_agents-1, max_calls)
                st.warning(f"With {min_agents-1} agents: Delay probability = {prev_metrics['delay_probability']*100:.1f}%")
            
            # Show performance with more agents
            if min_agents < max_calls:
                next_metrics = erlang_a_stationary_distribution(λ, μ, γ, min_agents+1, max_calls)
                st.info(f"With {min_agents+1} agents: Delay probability = {next_metrics['delay_probability']*100:.1f}%")
            
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")

if __name__ == "__main__":
    main()