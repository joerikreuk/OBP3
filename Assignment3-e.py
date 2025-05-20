import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus

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

def solve_shift_scheduling(agent_hours_needed):
    """
    Solves the shift scheduling problem using ILP
    Returns schedule and inefficiency metrics
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create all possible shifts (consecutive days)
    shifts_8hr = [
        ['Monday', 'Tuesday', 'Wednesday'],  # Mon-Wed
        ['Tuesday', 'Wednesday', 'Thursday'],
        ['Wednesday', 'Thursday', 'Friday'],
        ['Thursday', 'Friday', 'Saturday'],
        ['Friday', 'Saturday', 'Sunday']
    ]
    
    shifts_6hr = [
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday'],  # Mon-Thu
        ['Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        ['Wednesday', 'Thursday', 'Friday', 'Saturday'],
        ['Thursday', 'Friday', 'Saturday', 'Sunday']
    ]
    
    # Create the problem
    prob = LpProblem("Shift_Scheduling", LpMinimize)
    
    # Create decision variables
    x = {f"8hr_{i}": LpVariable(f"x8_{i}", 0, None, cat='Integer') 
         for i in range(len(shifts_8hr))}
    y = {f"6hr_{i}": LpVariable(f"y6_{i}", 0, None, cat='Integer') 
         for i in range(len(shifts_6hr))}
    
    # Objective function: minimize total scheduled hours
    prob += (8 * 3 * lpSum(x.values()) + 6 * 4 * lpSum(y.values()))
    
    # Constraints: meet or exceed required hours each day
    for day in days:
        # Sum of 8hr shifts covering this day
        eight_hr_cover = lpSum([x[f"8hr_{i}"] for i, shift in enumerate(shifts_8hr) if day in shift])
        # Sum of 6hr shifts covering this day
        six_hr_cover = lpSum([y[f"6hr_{i}"] for i, shift in enumerate(shifts_6hr) if day in shift])
        # Total scheduled hours must >= required hours
        prob += (8 * eight_hr_cover + 6 * six_hr_cover) >= agent_hours_needed[day]
    
    # Solve the problem
    prob.solve()
    
    # Collect results
    schedule = {
        '8hr_shifts': {f"Shift {i+1}": int(x[f"8hr_{i}"].varValue) for i in range(len(shifts_8hr))},
        '6hr_shifts': {f"Shift {i+1}": int(y[f"6hr_{i}"].varValue) for i in range(len(shifts_6hr))}
    }
    
    # Calculate scheduled hours per day
    scheduled_hours = {day: 0 for day in days}
    for i, shift in enumerate(shifts_8hr):
        count = int(x[f"8hr_{i}"].varValue)
        for day in shift:
            scheduled_hours[day] += 8 * count
    for i, shift in enumerate(shifts_6hr):
        count = int(y[f"6hr_{i}"].varValue)
        for day in shift:
            scheduled_hours[day] += 6 * count
    
    # Calculate inefficiency
    inefficiency = {
        day: (scheduled_hours[day] - agent_hours_needed[day]) / agent_hours_needed[day] 
        for day in days
    }
    
    total_scheduled = sum(scheduled_hours.values())
    total_required = sum(agent_hours_needed.values())
    total_inefficiency = (total_scheduled - total_required) / total_required
    
    return {
        'status': LpStatus[prob.status],
        'schedule': schedule,
        'scheduled_hours': scheduled_hours,
        'daily_inefficiency': inefficiency,
        'total_inefficiency': total_inefficiency,
        'total_scheduled_hours': total_scheduled,
        'total_required_hours': total_required
    }

def main():
    # Parameters (replace with your actual requirements)
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
        'Sunday': 1302.4,
    }
    
    # Calculate agent hours needed (from previous step)
    μ = 1 / aht
    γ = 1 / patience
    max_delay_prob = 1 - service_level
    
    agent_hours_needed = {}
    for day, volume in weekly_volume.items():
        λ = volume / (opening_hours * 60)
        min_agents = find_min_agents(λ, μ, γ, max_delay_prob)
        agent_hours_needed[day] = min_agents * opening_hours
    
    # Solve the shift scheduling problem
    solution = solve_shift_scheduling(agent_hours_needed)
    
    # Display results
    print("=== Agent Hours Required ===")
    print(pd.DataFrame.from_dict(agent_hours_needed, orient='index', columns=['Hours Needed']).to_string())
    
    print("\n=== Optimal Shift Schedule ===")
    print("8-hour shifts (3 consecutive days):")
    for shift, count in solution['schedule']['8hr_shifts'].items():
        if count > 0:
            print(f"{shift}: {count} teams")
    
    print("\n6-hour shifts (4 consecutive days):")
    for shift, count in solution['schedule']['6hr_shifts'].items():
        if count > 0:
            print(f"{shift}: {count} teams")
    
    print("\n=== Scheduled Hours ===")
    scheduled_df = pd.DataFrame({
        'Required': agent_hours_needed,
        'Scheduled': solution['scheduled_hours'],
        'Inefficiency': solution['daily_inefficiency']
    })
    print(scheduled_df.to_string(float_format="{:,.2f}".format))
    
    print("\n=== Summary ===")
    print(f"Total Required Hours: {solution['total_required_hours']:,.1f}")
    print(f"Total Scheduled Hours: {solution['total_scheduled_hours']:,.1f}")
    print(f"Total Inefficiency: {solution['total_inefficiency']:.2%}")

if __name__ == "__main__":
    main()