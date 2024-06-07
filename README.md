import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define constants
M = 0.529
Ks = 100
Cv = 10
k = 1
RL = 10
Cp = 1
u1 = 1
u2 = 0

# Define the state space model
def state_space(t, state):
    x1, x2, x3 = state
    dx1_dt = x2
    dx2_dt = 1/M * (u1 + u2 - k * x3 - Cv * x2 - Ks * x1)
    Vt = 1/Cp * x3
    dx3_dt = Vt - RL * x3
    return [dx1_dt, dx2_dt, dx3_dt]

# Initial conditions
initial_state = [0, 0, 0]

# Time span
t_span = (0, 10)  # simulate for 10 seconds
t_eval = np.linspace(0, 10, 1000)

# Solve the differential equations
solution = solve_ivp(state_space, t_span, initial_state, t_eval=t_eval)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(solution.t, solution.y[0])
plt.title('Displacement (x1) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')

plt.subplot(3, 1, 2)
plt.plot(solution.t, solution.y[1])
plt.title('Velocity (x2) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')

plt.subplot(3, 1, 3)
plt.plot(solution.t, solution.y[2])
plt.title('Current (x3) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')

plt.tight_layout()
plt.show()
