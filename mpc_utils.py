import numpy as np
from numba import jit, float64, int64

# np.polyval(coff, init_state[0])

@jit(float64[:](float64[:], float64[:], float64[:], float64, int64), nopython=True)
def model(inputs, init_state, coff, dt = 0.1, L = 3):
    final_state = np.array([0.0] * 6)
    final_state[0] = init_state[0]  + init_state[3]*np.cos(init_state[2])*dt
    final_state[1]  = init_state[1]  + init_state[3]*np.sin(init_state[2])*dt
    final_state[2] = init_state[2] + (init_state[3]/L)*inputs[0]*dt
    final_state[3]  = init_state[3]  + inputs[1]*dt
    th_des = np.arctan(coff[2] + 2*coff[1]*init_state[0] + 3*coff[0]*init_state[0]**2)
    final_state[4] = coff[0] * init_state[0] ** 2 + coff[1] * init_state[0] + coff[2] - init_state[1] + (init_state[3]*np.sin(init_state[5])*dt)
    final_state[5] = init_state[2] - th_des + ((init_state[3]/L)*inputs[0]*dt)
    return final_state

@jit(float64[:](float64[:], float64[:], float64, int64), nopython=True)
def simp_model(inputs, init_state, dt = 0.08, L = 3):
    final_state = np.array([0.0] * 4)
    final_state[0] = init_state[0] + init_state[3]*np.cos(init_state[2])*dt
    final_state[1] = init_state[1] + init_state[3]*np.sin(init_state[2])*dt
    final_state[2] = init_state[2] + (init_state[3]/L)*inputs[0]*dt
    final_state[3] = init_state[3] + inputs[1]*dt
    return final_state

def generate_ref(x_c, x_g, h, current_speed, dt = 0.08, bound = 13, max_acc = 3.2, P_gain = 1.00):
    """ generates waypoints
    x_c   - current position
    x_g   - goal position
    h     - horizon for waypoints
    v     - current_speed
    dt    - time step
    kp    - propotional gain
    bound - the maximum absolute speed 
    """
    xs = []
    
    for i in range(h):
        diff = x_g - x_c
        speed = P_gain * abs(np.linalg.norm(diff))
        nominal_speed = current_speed + max_acc * dt * (i + 1)
        if speed > nominal_speed: speed = nominal_speed
        if speed > bound: speed = bound
        v = speed * diff/(np.linalg.norm(diff))
        mag_v = np.linalg.norm(v)
                                
        x_c = x_c + dt * v
        xs.append([x_c[0], x_c[1], mag_v])
        
    return np.array(xs)