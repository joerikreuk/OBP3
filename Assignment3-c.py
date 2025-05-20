import numpy as np
import pandas as pd

def erlang_a_stationary_distribution(λ, μ, γ, s, max_calls=1000):
    """Calculate stationary distribution for Erlang-A call center model"""
    Q = np.zeros((max_calls + 1, max_calls + 1))
    
    for i in range(max_calls + 1):
        if i < max_calls:
            Q[i, i+1] = λ
        
        if i > 0:
            if i <= s:
                Q[i, i-1] = i * μ
            else:
                Q[i, i-1] = s * μ + (i - s) * γ
        
        Q[i, i] = -np.sum(Q[i, :])
    
    A = Q.T
    A[-1, :] = 1
    b = np.zeros(max_calls + 1)
    b[-1] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return {
        'delay_probability': sum(pi[s+1:]),
        'agent_utilization': sum(min(i, s)/s * pi[i] for i in range(1, max_calls+1))
    }

def find_min_agents(λ, μ, γ, max_delay_prob, max_calls=1000):
    """Find minimum agents needed by incremental search"""
    s = 1
    while True:
        metrics = erlang_a_stationary_distribution(λ, μ, γ, s, max_calls)
        if metrics['delay_probability'] <= max_delay_prob or s >= max_calls:
            return s
        s += 1

def calculate_agent_requirements(weekly_volume, service_level, aht, patience, opening_hours):
    """
    Calculate agent requirements for a week
    
    Parameters:
    weekly_volume - dict with daily volumes {'Monday': X, ...}
    service_level - target (e.g., 0.60 for 60/0)
    aht - average handling time in minutes
    patience - average patience in minutes
    opening_hours - daily operating hours
    """
    μ = 1 / aht  # Service rate (calls per minute)
    γ = 1 / patience  # Abandonment rate (calls per minute)
    max_delay_prob = 1 - service_level
    
    daily_results = {}
    
    for day, volume in weekly_volume.items():
        # Convert daily volume to calls per minute
        λ = volume / (opening_hours * 60)
        
        # Find minimum agents needed
        min_agents = find_min_agents(λ, μ, γ, max_delay_prob)
        
        daily_results[day] = {
            'volume': volume,
            'agents_required': min_agents,
            'agent_hours': min_agents * opening_hours
        }
    
    return pd.DataFrame.from_dict(daily_results, orient='index')

# Example usage with sample data
if __name__ == "__main__":
    # Parameters
    service_level = 0.60  # 60/0 service level
    aht = 5  # minutes
    patience = 10  # minutes
    opening_hours = 14  # hours per day
    
    # Sample forecast for week 260 (replace with actual forecast)
    weekly_volume = {
        'Monday': 1770.1,
        'Tuesday': 4104.4,
        'Wednesday': 2598.1,        
        'Thursday': 3174.6,
        'Friday': 3133.6,
        'Saturday': 1802.0,
        'Sunday': 1302.4
    }
    
    # Calculate requirements
    results = calculate_agent_requirements(weekly_volume, service_level, aht, patience, opening_hours)
    
    # Add summary row
    totals = pd.DataFrame({
        'volume': results['volume'].sum(),
        'agents_required': '',
        'agent_hours': results['agent_hours'].sum()
    }, index=['Total'])
    
    results = pd.concat([results, totals])
    
    # Display results
    print("Agent Requirements for Week 260:")
    print(results.to_markdown(floatfmt=".1f"))
    
    print(f"\nTotal Agent Hours Required: {results.loc['Total','agent_hours']:.1f}")