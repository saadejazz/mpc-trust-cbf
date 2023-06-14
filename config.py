# system configurations - only needed for CARLA
CARLA_ROOT = "C:/Users/Saad/Desktop/CARLA_Latest/WindowsNoEditor/PythonAPI/carla"

# model configurations
SMATO_MODEL = "trusty/models/smato_mobilenet_v2m"

# algorithm configurations
fluctuation_sensitvity = 0.25

# beta for momentum in trust - beta corresponds to proportion of history to take
beta_smato = 0.50
beta_fluc = 0.80

# trust based on eye contact detection is monotonically increasing, with the rate as follows
eye_rate = 0.10

# trust aggregation - must add to 1
alpha_trust = {
    "smato": 0.40,
    "eye":  0.50,
    "fluc": 0.10
}

# final trust dynamics
# set to non-zero if want to model as a monotonically increasing function
# trust_k+1 = trust_k + f_trust_inc * T_k+1
# in this case, initial_trust_ratio is the percentage of initial trust everyone starts with
# trust_0 = initial_trust_ratio * T_0
f_trust_inc = 0.075
initial_trust_ratio = 0.8