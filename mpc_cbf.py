import cutils
import numpy as np
from scipy.optimize import minimize
from mpc_utils import model, generate_ref, simp_model
from trusty import init_predictor
import timeit


# state variables order
# x
# y
# th
# v
# cte
# eth

# input variables order
# steer angle
# acceleration

init_state = np.array([0.0] * 6)
class Controller(object):
    def __init__(self, start_point, end_point, P = 7, R = 2.7):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._vel_x              = 0
        self._vel_y              = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._current_frame      = 0
        self._set_steer          = 0
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self.STEP_TIME           = 0.01
        self.R                   = R
        self.P                   = P
        self.trusts             = []
        self.obs_pos             = np.array([])
        self.obs_vel             = np.array([])
        self.snaptime            = 0
        self.update_waypoints(start_point, end_point, 0.0)
        self.trusty = init_predictor()
        self.trusty.predict(["resources/init_pic.png"])

    def update_values(self, x, y, yaw, speed, timestamp, vel, frame, snaptime):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._vel_x             = vel.x
        self._vel_y             = vel.y
        self._current_frame     = frame
        self.snaptime           = snaptime
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, x_c, x_g, current_speed):
        self._waypoints = generate_ref(x_c, x_g, 20, current_speed)

    @staticmethod
    def calculate_gamma(trust, gamma_lim = 0.5, delta = 2.7):
        '''Converts trust values to gamma for cbf constraints
        '''
        return 1/((1/gamma_lim) + delta * (np.exp((-1 * trust)) - 1))

    def perceive(self, obs_pos, obs_vel):
        self.obs_pos = np.array(sorted(obs_pos, key = lambda x: x[0]))
        self.obs_vel = np.array(sorted(obs_vel, key = lambda x: x[0]))

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def map_coord_2_Car_coord(self, x, y, yaw, waypoints):  
    
        wps = np.squeeze(waypoints)
        wps_x = wps[:,0]
        wps_y = wps[:,1]

        num_wp = wps.shape[0]
        
        ## create the Matrix with 3 vectors for the waypoint x and y coordinates w.r.t. car 
        wp_vehRef = np.zeros(shape=(3, num_wp))
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
                
        wp_vehRef[0,:] = cos_yaw * (wps_x - x) - sin_yaw * (wps_y - y)
        wp_vehRef[1,:] = sin_yaw * (wps_x - x) + cos_yaw * (wps_y - y)        

        return wp_vehRef   

    def update_trust(self, store_loc):
        trust_data = self.trusty.predict([store_loc])
        trust_data = sorted(trust_data, key = lambda x: x["keypoints"][0])
        self.trusts = [trust["trust"] for trust in trust_data]

    def update_controls(self):
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        self.vars.create_var('x_previous', 0.0)
        self.vars.create_var('y_previous', 0.0)
        self.vars.create_var('th_previous', 0.0)
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('cte_previous', 0.0)
        self.vars.create_var('eth_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('snaptime', 0.0)
        self.vars.create_var('prev_input', np.zeros(2*self.P))
        
        ## step time ##
        self.STEP_TIME = t - self.vars.t_previous
       
        ## init geuss ##
        x0 = self.vars.prev_input

        ## cost function weights ##
        cte_W = 25
        eth_W = 50
        v_W = 60
        st_rate_W = 200
        acc_rate_W = 10
        st_W = 100
        acc_W = 1

        ## input bounds ##
        b1 = (-1.22, 1.22)
        b2 = (0.0, 1.0)
        bnds = [b1] * self.P + [b2] * self.P
        assert(len(bnds) == 2 * self.P)
        
        wps_vehRef = self.map_coord_2_Car_coord(x, y, yaw, waypoints)
        wps_vehRef_x = wps_vehRef[0,:]
        wps_vehRef_y = wps_vehRef[1,:]

        ## find COFF of the polynomial ##
        coff = np.polyfit(wps_vehRef_x, wps_vehRef_y, 3)
        v_ref = v_desired

        if self._start_control_loop:
            def objective(x):
                u = np.array([0.0] * 2)
                Error = 0
                global init_state
                init_state_1 = init_state
                for i in range(self.P):
                    u[0] = x[i - 1]
                    u[1] = x[i + self.P]
                    
                    next_state = model(u, init_state_1, coff, dt = self.STEP_TIME, L = 3)
                    if i == 0 :
                        Error += cte_W*next_state[4]**2 + eth_W*next_state[5]**2 + v_W*(next_state[3] - v_ref)**2 \
                                + st_W*u[0]**2 + acc_W*u[1]**2
                    else:
                        Error += cte_W*next_state[4]**2 + eth_W*next_state[5]**2 + v_W*(next_state[3] - v_ref)**2 \
                                + st_rate_W*(u[0] - x[i-1])**2 + acc_rate_W*(u[1] - x[i + self.P - 1])**2 \
                                + st_W*(u[0])**2 + acc_W*(u[1])**2
                    init_state_1 = next_state
                return Error
            
            def cbf(x, obs_idx, p):
                u = np.array([0.0] * 2)
                next_state = np.array([0.0] * 4)
                next_state[0] = self._current_x
                next_state[1] = self._current_y
                next_state[2] = self._current_yaw
                next_state[3] = self._current_speed

                X_obs = self.obs_pos[obs_idx, :] + self.STEP_TIME * p * self.obs_vel[obs_idx, :]

                if p == 0:
                    X = np.array([next_state[0], next_state[1]])

                for i in range(p + 1):
                    u[0] = x[i - 1] 
                    u[1] = x[i + self.P]
                    next_state = simp_model(u, next_state, dt = self.STEP_TIME, L = 3)
                    if i == p - 1:
                        X = np.array([next_state[0], next_state[1]])
                    elif i == p:
                        X_1 = np.array([next_state[0], next_state[1]])

                if len(self.trusts) <= obs_idx: gamma = self.calculate_gamma(0.5)
                else: gamma = self.calculate_gamma(self.trusts[obs_idx]) 

                return np.sum((X_1 - X_obs) ** 2) - self.R*self.R + (gamma * self.STEP_TIME - 1)\
                       * (np.sum((X - X_obs) ** 2) - self.R*self.R)

            CarRef_x = CarRef_y = CarRef_yaw = 0.0
            cte = np.polyval(coff, CarRef_x) - CarRef_y

            # get orientation error from fit ( Since we are trying a 3rd order poly, then, f' = a1 + 2*a2*x + 3*a3*x2)
            # in this case and since we moved our reference sys to the Car, x = 0 and also yaw = 0
            yaw_err = CarRef_yaw - np.arctan(coff[1])

            # compensate for the latency by "predicting" what would be the state after the latency period.
            latency = 0.1

            # Predict the state; px, py and psi wrt car are all 0.
            init_state[0] = v * latency
            init_state[1] = 0
            init_state[2] = -v * self._set_steer * latency / 3
            init_state[3] = v + (v - self.vars.v_previous)/self.STEP_TIME * latency
            init_state[4] = cte + v * np.sin(yaw_err) * latency
            init_state[5] = yaw_err + init_state[2]

            cstr = []
            for i in range(self.obs_pos.shape[0]):
                for p in range(self.P):
                    cstr.append({
                        "type": "ineq",
                        "fun": cbf,
                        "args": (i, p, ),
                    })

            # start = timeit.default_timer()
            solution = minimize(objective, x0, method = 'SLSQP', bounds = bnds, constraints = cstr)
            # solution = minimize_ipopt(objective, x0, bounds = bnds, constraints = cstr)
            # print(timeit.default_timer() - start)

            u = solution.x
            self.vars.prev_input = u
            steer_output = u[0]

            if u[self.P] < 0 :
                brake_output = u[self.P]
            else:
                throttle_output = u[self.P]
            

            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        self.vars.t_previous = t  # Store timestamp  to be used in next step
        self.vars.v_previous = v