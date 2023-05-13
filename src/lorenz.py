#!/usr/bin/env python3
"""
Animate the evolution of a Lorenz system in Matplotlib
NOTE: the purpose of this exercise is the visualization, NOT the simulation.
A Runge-Kutta integration would very probably be more appropriate for a
chaotic dynamical system such as this one; this is elided here for clarity. 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy._typing import NDArray

DT = 0.01

def lorenz_system(vec: NDArray, sigma=10, rho=28, beta=2.667):
    """
    The system function for the lorenz system

    vec: the system state
    returns: the derivative of the system state 
    """
    x, y, z = vec[0], vec[1], vec[2]
    x_dot = sigma*(y - x)
    y_dot = rho*x - y - x*z
    z_dot = x*y - beta*z
    return np.array([x_dot, y_dot, z_dot])

def run_lorenz(n_steps: int, x0=np.array([0, 1, 1.05])):
    """
    Runs a simulation of the Lorenz system for a number of steps

    n_steps: number of steps
    x0: initial conditions 
    """
    state_arr = np.empty((n_steps, 3))
    state_arr[0,:] = x0
    for i in range(n_steps-1):
        state_arr[i+1,:] = state_arr[i] + DT*lorenz_system(state_arr[i,:])
    return state_arr

def update_lorenz(num: int, states: NDArray, line, rtf: float = 1):
    """
    num: the iteration number
    states: the states of the system as found in run_lorenz
    line: the line artist
    rtf: the real time factor (set to >1 to go faster) 
    """
    # set_data expects a 2xn, not an nx2
    # for 3d plots, the z dimension must be set separately
    num = int(rtf*num)
    line.set_data(states[:num, :2].T)
    line.set_3d_properties(states[:num, 2])

if __name__ == "__main__":
    n_steps = 2000
    rtf = 5
    lorenz_states = run_lorenz(n_steps)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim3d=(-20, 20), xlabel='X')
    ax.set(ylim3d=(-25, 35), ylabel='Y')
    ax.set(zlim3d=(0, 60), zlabel='Z')
    
    line = ax.plot([], [], [])[0]
    # interval is delay bw frames in ms
    ani = animation.FuncAnimation(
        fig, update_lorenz, int(n_steps/rtf), fargs=(lorenz_states, line, rtf), interval=50
    )

    ani.save('out/lorenz.gif')
    # plt.show()
