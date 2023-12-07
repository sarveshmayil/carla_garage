from jax import jit, jacfwd, jacrev, hessian, lax
import jax.numpy as np
from jax.scipy.special import logsumexp
import jax

np.set_printoptions(precision=3)

# from jax.config import config
# config.update("jax_enable_x64", True)
import sys
import numpy as onp
onp.set_printoptions(threshold=sys.maxsize)
import pickle
import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = [5, 5]

from tqdm.auto import tqdm

import time

from car_env_for_MPC import *

from scipy.integrate import solve_ivp

import os

DT = 0.1# [s] delta time step, = 1/FPS_in_server
N_X = 6
N_U = 2
TIME_STEPS = 50

@jit
def continuous_dynamics(state, action):
    #state = [X, u, Y, v, PSI, r]
    #action = [delta_f, F_x]

    #Carla Wheel Physics
    # friction coeff: 3.5, long_stiff: 3000.0, lat_stiff: 20.0, lat_stiff_max_load: 3.0
    # wheelbase: 3, track width: 1.67
    Nw=2 #Number wheels
    f=3.5#friction
    a=1.35
    b=1.45
    # By=0.27
    By = 20.0
    Cy=1.2
    # Dy=0.7
    Dy = 3
    Ey=-1.6
    Shy=0
    Svy=0
    m=1845
    # Iz = 2667
    L = 3 #Wheelbase
    Iz=0.5*m*(L/2)**2

    g=9.806

    delta_f = action[0]
    F_x = action[1]

    a_f = np.rad2deg(delta_f-np.arctan2(state[3] + a*state[5], state[1]))
    a_r = np.rad2deg(-np.arctan2(state[3] - b*state[5], state[1]))

    phi_yf=(1-Ey)*(a_f+Shy)+(Ey/By)*np.arctan(By*(a_f+Shy))
    phi_yr=(1-Ey)*(a_r+Shy)+(Ey/By)*np.arctan(By*(a_r+Shy))

    F_zf=b/(a+b)*m*g
    F_yf=F_zf*Dy*np.sin(Cy*np.arctan(By*phi_yf))+Svy

    F_zr=a/(a+b)*m*g
    F_yr=F_zr*Dy*np.sin(Cy*np.arctan(By*phi_yr))+Svy

    # F_total=np.sqrt((Nw*F_x)**2+(F_yr**2))
    # F_max=0.7*m*g
    # if F_total>F_max:
    #     F_x=F_max/F_total*F_x
    #     F_yr=F_max/F_total*F_yr


    dzdt = [state[1]*np.cos(state[4]) - state[3]*np.sin(state[4]), 
            (-f*m*g+Nw*F_x-F_yf*np.sin(delta_f))/m+state[3]*state[5], 
            state[1]*np.sin(state[4]) + state[3]*np.cos(state[4]), 
            (F_yf*np.cos(delta_f) + F_yr)/m-state[1]*state[5], 
            state[5],
            (F_yf*a*np.cos(delta_f)-F_yr*b)/Iz]
    
    return np.array(dzdt)
@jit
def discrete_dynamics(state, u):
    return state + continuous_dynamics(state, u)*DT

@jit
def rollout(x0, u_trj):
    x_final, x_trj = jax.lax.scan(rollout_looper, x0, u_trj)
    return np.vstack((x0, x_trj))
    
@jit
def rollout_looper(x_i, u_i):
    x_ip1 = discrete_dynamics(x_i, u_i)
    return x_ip1, x_ip1

# TODO: remove length if weighing func is not needed
@jit
def distance_func(x, route):
    x, ret = lax.scan(distance_func_looper, x, route)
    return -logsumexp(ret)

@jit
def distance_func_looper(input_, p):
    global dp
    
    delta_x = input_[0]-p[0]
    delta_y = input_[2]-p[1]

    return input_, -(delta_x**2.0 + delta_y**2.0)/(1.0*dp**2.0)


@jit
def cost_1step(state, action, route, goal_speed = 8.):
    '''
    state has shape [N_x,]
    action has shape [N_u,]
    route has shape [W, N_x]
    '''
    global TIME_STEPS_RATIO


    speed =np.sqrt(np.square(state[1]) + np.square(state[3]))

    c_position = distance_func(state, route)
    c_speed = np.square(speed-goal_speed)
    # c_control = jnp.square(action[0]/0.7)
    # c_control = jnp.square(action[1]/5000)
    c_control = np.sqrt(np.square(action[0]/0.5) + np.square(action[1]/5000))

    # c_control = action[0]/0.5 + action[1]/500
    # return (0.*c_position + 0.0005*c_speed + 0.*c_control)/TIME_STEPS_RATIO #~2.3
    # return (.001*c_position + 0.*c_speed + 0.*c_control)/TIME_STEPS_RATIO #~1.29
    # return (0.*c_position + 0.*c_speed + 0.025*c_control)/TIME_STEPS_RATIO #~1.7
    # return (0.003*c_position + 0.00025*c_speed + 0.025*c_control)/TIME_STEPS_RATIO #~7 on init
    return (0.03*c_position + 0.00025*c_speed + 0.025*c_control)/TIME_STEPS_RATIO #~7 on init
    # return (0.6*c_position + 0.0025*c_speed + 0.04*c_control)/TIME_STEPS_RATIO #~7 on init
    # return 0.03*c_position

@jit
def cost_final(state, route): 
    '''
    state has shape [N_x,]
    route has shape [W, N_x]
    '''
    global TARGET_RATIO
    c_position = np.square(state[0]-route[-1,0]) + np.square(state[2]-route[-1,1])
    c_speed =np.sqrt(np.square(state[1]) + np.square(state[3]))

    # return (0.003*c_position/(TARGET_RATIO**2) + 0.*c_speed)*1
    return 0.06*(c_position/(TARGET_RATIO**2) + 0.0*c_speed)*1

@jit
def cost_trj(x_trj, u_trj, route):
    total = 0.
    total, x_trj, u_trj, route = jax.lax.fori_loop(0, TIME_STEPS-1, cost_trj_looper, [total, x_trj, u_trj, route])
    total += cost_final(x_trj[-1], route)
    
    return total

# XXX: check if the cost_1step needs `target`
@jit
def cost_trj_looper(i, input_):
    total, x_trj, u_trj, route = input_
    total += cost_1step(x_trj[i], u_trj[i], route)
    
    return [total, x_trj, u_trj, route]

def derivative_init():
    jac_l = jit(jacfwd(cost_1step, argnums=[0,1]))
    hes_l = jit(hessian(cost_1step, argnums=[0,1]))
    jac_l_final = jit(jacfwd(cost_final))
    hes_l_final = jit(hessian(cost_final))
    jac_f = jit(jacfwd(discrete_dynamics, argnums=[0,1]))
    
    return jac_l, hes_l, jac_l_final, hes_l_final, jac_f

jac_l, hes_l, jac_l_final, hes_l_final, jac_f = derivative_init()

@jit
def derivative_stage(x, u, route): # x.shape:(5), u.shape(3)
    global jac_l, hes_l, jac_f
    l_x, l_u = jac_l(x, u, route)
    (l_xx, l_xu), (l_ux, l_uu) = hes_l(x, u, route)
    f_x, f_u = jac_f(x, u)

    return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

@jit
def derivative_final(x, target):
    global jac_l_final, hes_l_final
    l_final_x = jac_l_final(x, target)
    l_final_xx = hes_l_final(x, target)

    return l_final_x, l_final_xx

@jit
def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    Q_x = l_x + f_x.T@V_x
    Q_u = l_u + f_u.T@V_x

    Q_xx = l_xx + f_x.T@V_xx@f_x
    Q_ux = l_ux + f_u.T@V_xx@f_x
    Q_uu = l_uu + f_u.T@V_xx@f_u

    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

@jit
def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = - Q_uu_inv@Q_u
    K = - Q_uu_inv@Q_ux

    return k, K

@jit
def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
    V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K

    return V_x, V_xx

@jit
def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T@k - 0.5 * k.T@Q_uu@k

@jit
def forward_pass(x_trj, u_trj, k_trj, K_trj):
    u_trj = np.arcsin(np.sin(u_trj))
    
    x_trj_new = np.empty_like(x_trj)
    x_trj_new = x_trj_new.at[0].set(x_trj[0])
    u_trj_new = np.empty_like(u_trj)
    
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = lax.fori_loop(
        0, TIME_STEPS-1, forward_pass_looper, [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]
    )

    return x_trj_new, u_trj_new

@jit
def forward_pass_looper(i, input_):
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = input_
    
    u_next = u_trj[i] + k_trj[i] + K_trj[i]@(x_trj_new[i] - x_trj[i])
    u_trj_new = u_trj_new.at[i].set(u_next)

    x_next = discrete_dynamics(x_trj_new[i], u_trj_new[i])
    x_trj_new = x_trj_new.at[i+1].set(x_next)
    
    return [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]

@jit
def backward_pass(x_trj, u_trj, regu, target):
    k_trj = np.empty_like(u_trj)
    K_trj = np.empty((TIME_STEPS-1, N_U, N_X))
    expected_cost_redu = 0.
    V_x, V_xx = derivative_final(x_trj[-1], target)
     
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target = lax.fori_loop(
        0, TIME_STEPS-1, backward_pass_looper, [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target]
    )
        
    return k_trj, K_trj, expected_cost_redu


@jit
def backward_pass_looper(i, input_):
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target = input_
    n = TIME_STEPS-2-i
    
    l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivative_stage(x_trj[n], u_trj[n], target)
    Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
    Q_uu_regu = Q_uu + np.eye(N_U)*regu
    k, K = gains(Q_uu_regu, Q_u, Q_ux)
    k_trj = k_trj.at[n].set(k)
    
    K_trj = K_trj.at[n].set(K)
    V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
    expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    
    return [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target]

@jit
def run_ilqr_main(x0, u_trj, target):
    global jac_l, hes_l, jac_l_final, hes_l_final, jac_f
    
    max_iter=300
    regu = np.array(100.)
    
    x_trj = rollout(x0, u_trj)
    cost_trace = np.zeros((max_iter+1))
    cost_trace = cost_trace.at[0].set(cost_trj(x_trj, u_trj, target))

    x_trj, u_trj, cost_trace, regu, target = lax.fori_loop(
        1, max_iter+1, run_ilqr_looper, [x_trj, u_trj, cost_trace, regu, target]
    )
    
    return x_trj, u_trj, cost_trace

@jit
def run_ilqr_looper(i, input_):
    x_trj, u_trj, cost_trace, regu, target = input_
    k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu, target)
    x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)
    
    total_cost = cost_trj(x_trj_new, u_trj_new, target)
    
    x_trj, u_trj, cost_trace, regu = lax.cond(
        pred = (cost_trace[i-1] > total_cost),
        true_operand = [i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new, regu],
        true_fun = run_ilqr_true_func,
        false_operand = [i, cost_trace, x_trj, u_trj, regu],
        false_fun = run_ilqr_false_func,
    )
    
    max_regu = 10000.0
    min_regu = 0.01
    
    regu += jax.nn.relu(min_regu - regu)
    regu -= jax.nn.relu(regu - max_regu)

    return [x_trj, u_trj, cost_trace, regu, target]

@jit
def run_ilqr_true_func(input_):
    i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new, regu = input_
    
    cost_trace = cost_trace.at[i].set(total_cost)
    x_trj = x_trj_new
    u_trj = u_trj_new
    regu *= 0.7
    
    return [x_trj, u_trj, cost_trace, regu]

@jit
def run_ilqr_false_func(input_):
    i, cost_trace, x_trj, u_trj, regu = input_
    
    cost_trace = cost_trace.at[i].set(cost_trace[i-1] )

    regu *= 2.0
    
    return [x_trj, u_trj, cost_trace, regu]


TIME_STEPS = 60

# NOTE: Set dp to be the same as carla
dp = 1 # same as waypoint interval

onp.random.seed(1)


TIME_STEPS_RATIO = TIME_STEPS/50
# TARGET_RATIO = np.linalg.norm(target[-1]-target[0])/(3*np.pi)
TARGET_RATIO = FUTURE_WAYPOINTS_AS_STATE*dp/(6*np.pi) # TODO: decide if this should be determined dynamically





def continuous_dynamics_np(t, state):
    #state = [X, u, Y, v, PSI, r, steer, thrust]
    #action = [delta_f, F_x]

    #Carla Wheel Physics
    # friction coeff: 3.5, long_stiff: 3000.0, lat_stiff: 20.0, lat_stiff_max_load: 3.0
    # wheelbase: 3, track width: 1.67
    Nw=2.
    f=0.01
    Iz=2667
    a=1.35
    b=1.45
    By=0.27
    Cy=1.2
    Dy=0.7
    Ey=-1.6
    Shy=0.
    Svy=0.
    m=1400.
    g=9.806

    delta_f = state[6]
    F_x = state[7]

    a_f = onp.rad2deg(delta_f-onp.arctan2(state[3] + a*state[5], state[1]))
    a_r = onp.rad2deg(-onp.arctan2(state[3] - b*state[5], state[1]))

    phi_yf=(1-Ey)*(a_f+Shy)+(Ey/By)*onp.arctan(By*(a_f+Shy))
    phi_yr=(1-Ey)*(a_r+Shy)+(Ey/By)*onp.arctan(By*(a_r+Shy))

    F_zf=b/(a+b)*m*g
    F_yf=F_zf*Dy*onp.sin(Cy*onp.arctan(By*phi_yf))+Svy

    F_zr=a/(a+b)*m*g
    F_yr=F_zr*Dy*onp.sin(Cy*onp.arctan(By*phi_yr))+Svy

    F_total=np.sqrt((Nw*F_x)**2+(F_yr**2))
    F_max=0.7*m*g
    if F_total>F_max:
        F_x=F_max/F_total*F_x
        F_yr=F_max/F_total*F_yr


    dzdt = [state[1]*onp.cos(state[4]) - state[3]*onp.sin(state[4]), 
            (-f*m*g+Nw*F_x-F_yf*onp.sin(delta_f))/m+state[3]*state[5], 
            state[1]*onp.sin(state[4]) + state[3]*onp.cos(state[4]),    
            (F_yf*onp.cos(delta_f) + F_yr)/m- state[1]*state[5], 
            state[5],
            (F_yf*a*onp.cos(delta_f)-F_yr*b)/Iz,
            0, #Zero order hold input
            0] #Zero order hold input
    
    return onp.array(dzdt)

def main(waypoints):
    MPC_INTERVAL = 4
    init_vector = waypoints[0,:] - waypoints[1,:]
    init_yaw = np.arctan2(init_vector[1], init_vector[0])

    state = onp.zeros(6)
    state[4] = init_yaw
    state = np.array(state)


    time = [0]
    x_trajectory = [0]
    y_trajectory = [0]
    steers = []
    thrusts = []
    for k in tqdm(range(100)):
        # start = time.time()
        # state[2] += 0.01
        state = np.array(state)
        
        # u_trj = np.random.randn(TIME_STEPS-1, N_U)
        steer_sample = onp.random.randn(TIME_STEPS-1, 1) * 0.2
        thrust_sample = onp.random.randn(TIME_STEPS-1, 1) * 1000
        u_init = onp.hstack((steer_sample, thrust_sample))
        u_init = np.array(u_init)        
        waypoints = np.array(waypoints)

        x_trj, u_trj, cost_trace = run_ilqr_main(state, u_init, waypoints)
        print(cost_trace)
        for j in range(MPC_INTERVAL):
            steer = u_trj[j,0]
            thrust = u_trj[j,1]
            
            action = onp.array([steer, thrust])

            actual_state = onp.array(state)
            augmented_state = onp.hstack((actual_state, action))

            sol = solve_ivp(continuous_dynamics_np, [0, DT], augmented_state, atol= 1e-5, rtol = 1e-5)

            state = sol.y[:6, -1]
            steers.append(steer)
            thrusts.append(thrust)
            x_trajectory.append(state[0])
            y_trajectory.append(state[2])
            time.append(time[-1]+DT)

    fig, ax = plt.subplots(3)
    ax[0].scatter(waypoints[:,0], waypoints[:,1], label = "Waypoints", color = "red")
    ax[0].plot(x_trajectory, y_trajectory, label = "Trajectory", color = "blue")

    ax[1].scatter(time[:-1], thrusts)
    ax[2].scatter(time[:-1], steers)
    
    plt.show()



    # #Trying to get dynamics to match matlab    
    # state = onp.zeros(8)
    # state[1] = 8.9
    # t = onp.linspace(0, 5, 500)
    # dt = t[1]-t[0]
    # steer = onp.pi/6*onp.ones_like(t)
    # thrust = 1000*onp.ones_like(t)

    # x_trajectory = [0]
    # y_trajectory = [0]
    # for idx, t in enumerate(t):
    #     state[6] = steer[idx]
    #     state[7] = thrust[idx]
    #     sol = solve_ivp(continuous_dynamics_np, [t, t+dt], state)
    #     print(sol.y)
    #     state = sol.y[:, -1]
    #     x_trajectory.append(state[0])
    #     y_trajectory.append(state[2])
    
    # sol = solve_ivp(continuous_dynamics_np, [0, 5], state, atol= 1e-8, rtol = 1e-8)
    
    # fig, ax = plt.subplots()
    # ax.plot(x_trajectory, y_trajectory, label = "Trajectory", color = "blue")
    # # ax.plot(sol.y[0,:], sol.y[2,:], label = "Trajectory", color = "blue")

    # plt.show()

if __name__ == "__main__":
    waypoints = onp.array([[-48.674,  46.955],
                            [-48.672,  47.955],
                            [-48.669,  48.955],
                            [-48.666,  49.955],
                            [-48.663,  50.955],
                            [-48.66,   51.955],
                            [-48.657,  52.955],
                            [-48.655,  53.955],
                            [-48.652,  54.955],
                            [-48.649,  55.955],
                            [-48.646, 56.955],
                            [-48.643,  57.955],
                            [-48.641,  58.955],
                            [-48.638,  59.955],
                            [-48.635,  60.955],
                            [-48.632,  61.955],
                            [-48.629,  62.955],
                            [-48.627,  63.955],
                            [-48.624,  64.955],
                            [-48.621,  65.955],
                            [-48.618,  66.955],
                            [-48.615,  67.955],
                            [-48.612,  68.955],
                            [-48.61,   69.955],
                            [-48.607,  70.955],
                            [-48.604,  71.955],
                            [-48.601,  72.955],
                            [-48.598,  73.955],
                            [-48.596,  74.955],
                            [-48.593,  75.955],
                            [-48.59,   76.955],
                            [-48.587,  77.955],
                            [-48.584,  78.955],
                            [-48.582,  79.955],
                            [-48.579,  80.955],
                            [-48.576,  81.955],
                            [-48.573, 82.955],
                            [-48.57,   83.955],
                            [-48.567,  84.955],
                            [-48.565,  85.955],
                            [-48.562,  86.955],
                            [-48.559,  87.955],
                            [-48.556, 88.955],
                            [-48.553, 89.955],
                            [-48.551, 90.955],
                            [-48.548,  91.955],
                            [-48.545, 92.955],
                            [-48.542, 93.955],
                            [-48.539,  94.955],
                            [-48.536, 95.955]])
    waypoints -= waypoints[0, :]
    # print(waypoints)
    main(waypoints)
    
