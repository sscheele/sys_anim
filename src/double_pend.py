#!/usr/bin/env python3
"""
Animate a double pendulum system in Matplotlib 
This gives an example of how you can animate a system where you only need to
show a current state, rather than a full history - the system states need
not be stored or computed ahead of time
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from numpy._typing import NDArray

DT = 0.01

def system(state: NDArray, m=0.5, l=0.5, g=9.81):
    """
    The system function for the double pendulum 

    state: the system state
    returns: the derivative of the system state
    """
    theta1, theta2 = state[0], state[1]
    omega1, omega2 = state[2], state[3]

    theta_coeff = (6/(m*l*l))/(16 - (9*np.cos(theta1 - theta2)**2))
    theta1_dot = theta_coeff*(2*omega1 - 3*np.cos(theta1-theta2)*omega2)
    theta2_dot = theta_coeff*(8*omega2 - 3*np.cos(theta1-theta2)*omega1)

    omega_coeff = -0.5*m*l*l
    omega1_dot = omega_coeff*(theta1_dot*theta2_dot*np.sin(theta1-theta2) + \
        3*g/l*np.sin(theta1))
    omega2_dot = omega_coeff*(-theta1_dot*theta2_dot*np.sin(theta1-theta2) + \
        g/l*np.sin(theta2))
    
    return np.array([theta1_dot, theta2_dot, omega1_dot, omega2_dot])

def update_anim(num: int, state: List[NDArray], rects: List[Rectangle], rtf: int = 1, l=0.5):
    """
    num: the iteration number
    states: the states of the system as found in run_lorenz
    line: the line artist
    rtf: the real time factor (set to >1 to go faster, ints only) 
    """
    # set_data expects a 2xn, not an nx2
    # for 3d plots, the z dimension must be set separately
    frame = int(rtf*num)
    curr_state = state[0] 
    for _ in range(rtf):
        curr_state += DT*system(curr_state)
    arm1, arm2 = rects[0], rects[1]

    x1, y1 = l*np.sin(curr_state[0]), -l*np.cos(curr_state[0])
    arm1.set_xy((x1, y1))
    arm1.set_angle((180/np.pi)*curr_state[0])
    
    x2, y2 = x1 + l*np.sin(curr_state[1]), y1 - l*np.cos(curr_state[1])
    arm2.set_xy((x2, y2))
    arm2.set_angle((180/np.pi)*curr_state[1])

    # we overwrite state instead of returning out of the function, allowing
    # us to use the solution from this timestep next time
    state[0] = curr_state

if __name__ == "__main__":
    n_steps = 1000
    rtf = 3
    x = [np.array([np.pi/2, 0, 0, 0])]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(xlim=(-3, 3), xlabel='X')
    ax.set(ylim=(-3, 3), ylabel='Y')
    
    arm_rects = [Rectangle((0,0), 0.05, 0.5, fill=True) for _ in range(2)]
    for r in arm_rects:
        ax.add_patch(r)
    # interval is delay bw frames in ms
    ani = animation.FuncAnimation(
        fig, update_anim, int(n_steps/rtf), fargs=(x, arm_rects, rtf, 0.5), interval=50
    )

    # if the aspect ratio isn't set, the plot axis ratio will be
    # derived from the screen aspect ratio, making the rectangles appear
    # to distort as they rotate
    plt.gca().set_aspect('equal')
    ani.save('out/double_pend.gif')
    # plt.show()

