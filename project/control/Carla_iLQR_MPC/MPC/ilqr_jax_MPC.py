from jax import jit, jacfwd, jacrev, hessian, lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax

jnp.set_printoptions(precision=3)

# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as np

import pickle
import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = [5, 5]

from tqdm.auto import tqdm

import time

from car_env_for_MPC import *

import os

DT = 0.1# [s] delta time step, = 1/FPS_in_server
N_X = 6
N_U = 2
TIME_STEPS = 50
# MODEL_NAME = "bicycle_model_100000_v2_jax"
# MODEL_NAME = "bicycle_model_100ms_20000_v4_jax"
# model_path="../SystemID/model/net_{}.model".format(MODEL_NAME)
# NN_W1, NN_W2, NN_W3, NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))


@jit
def continuous_dynamics(state, action):
    #state = [X, u, Y, v, PSI, r]
    #action = [delta_f, F_x]
    Nw=2
    f=0.01
    Iz=2667
    a=1.35
    b=1.45
    By=0.27
    Cy=1.2
    Dy=0.7
    Ey=-1.6
    Shy=0
    Svy=0
    m=1847
    g=9.806

    delta_f = action[0]
    F_x = action[1]

    a_f = jnp.rad2deg(delta_f-jnp.arctan2(state[3] + a*state[5], state[1]))
    a_r = jnp.rad2deg(-jnp.arctan2(state[3] - b*state[5], state[1]))

    phi_yf=(1-Ey)*(a_f+Shy)+(Ey/By)*jnp.arctan(By*(a_f+Shy))
    phi_yr=(1-Ey)*(a_r+Shy)+(Ey/By)*jnp.arctan(By*(a_r+Shy))

    F_zf=b/(a+b)*m*g
    F_yf=F_zf*Dy*jnp.sin(Cy*jnp.arctan(By*phi_yf))+Svy

    F_zr=a/(a+b)*m*g
    F_yr=F_zr*Dy*jnp.sin(Cy*jnp.arctan(By*phi_yr))+Svy

    # F_total=np.sqrt((Nw*F_x)**2+(F_yr**2))
    # F_max=0.7*m*g

    # if F_total>F_max:
        
    #     F_x=F_max/F_total*F_x
    
    #     F_yr=F_max/F_total*F_yr


    dzdt = [state[1]*jnp.cos(state[4]) - state[3]*jnp.sin(state[4]), 
            (-f*m*g+Nw*F_x-F_yf*jnp.sin(delta_f))/m+state[3]*state[5], 
            state[2]*jnp.sin(state[4]) + state[3]*jnp.cos(state[4]), 
            (F_yf*jnp.cos(delta_f) + F_yr) * jnp.reciprocal(m-state[1]*state[5]), 
            state[5],
            (F_yf*a*jnp.cos(delta_f)-F_yr*b)/Iz]
    
    return jnp.array(dzdt)

@jit
def discrete_dynamics(state, u, dt = 0.1):
    '''
    state has shape [N_X,]
    u has shape [N_U,]
    '''
    return state + dt*continuous_dynamics(state, u)

@jit
def rollout(x0, u_trj):
    '''
    state has shape [N_X,]
    u has shape [T, N_U]
    '''
    x_final, x_trj = jax.lax.scan(rollout_looper, x0, u_trj)
    return jnp.vstack((x0, x_trj))
    
@jit
def rollout_looper(x_i, u_i):
    x_ip1 = discrete_dynamics(x_i, u_i)
    return x_ip1, x_ip1


@jit
def distance_func(x, route):
    '''
    state has shape [N_x,]
    route has shape [W, N_x]
    '''
    x, ret = jax.lax.scan(distance_func_looper, x, route)
    return -logsumexp(ret)

@jit
def distance_func_looper(input_, p):
    global dp
    
    delta_x = input_[0]-p[0]
    delta_y = input_[1]-p[1]

    return input_, -(delta_x**2.0 + delta_y**2.0)/(1.0*dp**2.0)

@jit
def cost_1step(state, action, route, goal_speed = 8.):
    '''
    state has shape [N_x,]
    action has shape [N_u,]
    route has shape [W, N_x]
    '''
    global TIME_STEPS_RATIO
    # R = jnp.diag(np.array([1., 1.]))
    # cost_weights = jnp.diag(np.array([1., 1., 1.]))

    speed =jnp.sqrt(jnp.square(state[1]) + jnp.square(state[3]))

    c_position = distance_func(state, route)
    c_speed = jnp.square(speed-goal_speed)
    # c_control = action.T@R@action

    # costs = jnp.array([c_position, c_speed, c_control])
    
    # return jnp.asarray(costs.T@cost_weights@costs)
    c_control = action[0]/0.5 + action[1]/500

    return (8.0*c_position + 2.0*c_speed + 0.05*c_control)/TIME_STEPS_RATIO

@jit
def cost_final(state, route): 
    '''
    state has shape [N_x,]
    route has shape [W, N_x]
    '''
    global TARGET_RATIO
    c_position = jnp.square(state[0]-route[-1,0]) + jnp.square(state[2]-route[-1,1])
    c_speed =jnp.sqrt(jnp.square(state[1]) + jnp.square(state[3]))

    return (c_position/(TARGET_RATIO**2) + 0.0*c_speed)*1

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
    '''
    x has shape [N_x,]
    u has shape [N_u,]
    route has shape [W, N_x]
    '''
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
    Q_uu_inv = jnp.linalg.inv(Q_uu)
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
    u_trj = jnp.arcsin(jnp.sin(u_trj))
    
    x_trj_new = jnp.empty_like(x_trj)
    #x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[0], x_trj[0])
    x_trj_new.at[0].set(x_trj[0])

    u_trj_new = jnp.empty_like(u_trj)
    
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = lax.fori_loop(
        0, TIME_STEPS-1, forward_pass_looper, [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]
    )

    return x_trj_new, u_trj_new

@jit
def forward_pass_looper(i, input_):
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = input_
    
    u_next = u_trj[i] + k_trj[i] + K_trj[i]@(x_trj_new[i] - x_trj[i])
    #u_trj_new = jax.ops.index_update(u_trj_new, jax.ops.index[i], u_next)
    u_trj_new.at[i].set(u_next)

    x_next = discrete_dynamics(x_trj_new[i], u_trj_new[i])
    #x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[i+1], x_next)
    x_trj_new.at[i+1].set(x_next)
    
    return [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]

@jit
def backward_pass(x_trj, u_trj, regu, target):
    k_trj = jnp.empty_like(u_trj)
    K_trj = jnp.empty((TIME_STEPS-1, N_U, N_X))
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
    Q_uu_regu = Q_uu + jnp.eye(N_U)*regu
    k, K = gains(Q_uu_regu, Q_u, Q_ux)
    #k_trj = jax.ops.index_update(k_trj, jax.ops.index[n], k)
    k_trj.at[n].set(k)
    #K_trj = jax.ops.index_update(K_trj, jax.ops.index[n], K)
    K_trj.at[n].set(K)
    V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
    expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    
    return [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target]

@jit
def run_ilqr_main(x0, u_trj, target):
    global jac_l, hes_l, jac_l_final, hes_l_final, jac_f
    
    max_iter=300
    regu = jnp.array(100.)
    
    x_trj = rollout(x0, u_trj)
    # cost_trace = jax.ops.index_update(
    #     jnp.zeros((max_iter+1)), jax.ops.index[0], cost_trj(x_trj, u_trj, target)
    # )
    cost_trace = jnp.zeros((max_iter+1))
    cost_trace.at[0].set(cost_trj(x_trj, u_trj, target))

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
    
    # cost_trace = jax.ops.index_update(
    #     cost_trace, jax.ops.index[i], total_cost 
    # )
    cost_trace.at[i].set(total_cost)

    x_trj = x_trj_new
    u_trj = u_trj_new
    regu *= 0.7
    
    return [x_trj, u_trj, cost_trace, regu]

@jit
def run_ilqr_false_func(input_):
    i, cost_trace, x_trj, u_trj, regu = input_
    
    # cost_trace = jax.ops.index_update(
    #     cost_trace, jax.ops.index[i], cost_trace[i-1] 
    # )
    cost_trace.at[i].set(cost_trace[i-1])
    regu *= 2.0
    
    return [x_trj, u_trj, cost_trace, regu]


TIME_STEPS = 60

# # NOTE: Set dp to be the same as carla
dp = 1 # same as waypoint interval

# np.random.seed(1)


TIME_STEPS_RATIO = TIME_STEPS/50
# # TARGET_RATIO = np.linalg.norm(target[-1]-target[0])/(3*np.pi)
TARGET_RATIO = FUTURE_WAYPOINTS_AS_STATE*dp/(6*jnp.pi) # TODO: decide if this should be determined dynamically


# carla init
env = CarEnv()
for i in range(1):
    state, waypoints = env.reset()
    # total_time = 0

    for k in tqdm(range(2000)):
        # start = time.time()

        # state[2] += 0.01
        state = jnp.array(state)
        
        # u_trj = np.random.randn(TIME_STEPS-1, N_U)
        steer_sample = np.random.randn(TIME_STEPS-1, 1) * 0.7
        thrust_sample = np.random.randn(TIME_STEPS-1, 1) * 2500 + 4000
        u_trj = np.hstack((steer_sample, thrust_sample))


        # u_trj[:,1] -= jnp.pi/2.5
        # u_trj[:,1] -= np.pi/8
        u_trj = jnp.array(u_trj)
        waypoints = jnp.array(waypoints)
        
        x_trj, u_trj, cost_trace = run_ilqr_main(state, u_trj, waypoints)
        # end = time.time()
        # if k > 1:
        #     total_time += end - start
        
        draw_planned_trj(env.world, np.array(x_trj), env.location_[2], color=(0, 223, 222))
        for j in range(MPC_INTERVAL):
            steer = u_trj[j,0]
            thrust = u_trj[j,1]

            # thrust = np.clip(thrust, -5000, 5000)
            # steering = jnp.sin(u_trj[j,0])
            # throttle = jnp.sin(u_trj[j,1])*0.5 + 0.5
            # brake = jnp.sin(u_trj[j,2])*0.5 + 0.5
            state, waypoints, done, _ = env.step(np.array([steer, thrust]))

        # tqdm.write("final estimated cost = {0:.2f} \n velocity = {1:.2f}".format(cost_trace[-1],state[2]))
        # if k > 1:
        #     tqdm.write("mean MPC calc time = {}".format(total_time/(k)))

        # if done:
        #     break

pygame.quit()

if VIDEO_RECORD:
    pass
    # os.system("ffmpeg -r 50 -f image2 -i Snaps/%05d.png -s {}x{} -aspect 16:9 -vcodec libx264 -crf 25 -y Videos/result.avi".
    #             format(RES_X, RES_Y))

# if __name__ == "__main__":
#     state = jnp.array([0., 0., 0., 0., 0., 0.])

#     state_trj = np.random.randn(TIME_STEPS-1, N_X)*1e-8    
#     u_trj = np.random.randn(TIME_STEPS-1, N_U)*1e-8

#     route = jnp.array([[0., 0.],
#                         [1., 0.],
#                         [2., 0.],
#                         [3., 0.],
#                         [4., 0.],
#                         [5., 0.]])

#     cost_trj(state_trj, u_trj, route)

