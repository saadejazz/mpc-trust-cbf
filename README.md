# MPC with trust-based CBF constraints

This repository contains work built on the concept of _system-to-human_ trust that focuses on estimating huamn inattention from perceived images in an attempt to measure their reliability for actively maintaining safety. This trust estimation is detailed in the sister repository [trusty](https://github.com/saadejazz/trusty). This repository uses estimated trust as a control aggressiveness parameter for CBF constraints in an MPC framework for autonomous driving simulated in CARLA. This work is ongoing and will be updated if needed.  

## Simulation Environment Setting for CARLA
An ego vehicle is simulated along with its interactions with the pedestrians in the surrounding. Two types of pedestrians are simulated. One type is posed so that the camera interprets smartphone usage and lack of eye contact. This pedestrian has low trust. The other pedestrian type simulated has a neutral pose with body facing the car, hence it is characterized as a high-trust pedestrian. The proposed algorithm navigates in such a fashion that it tries to be closer to trustworthy pedestrians and maintains more distance, relatively from untrustworthy ones.

![CARLA_main_example](https://github.com/saadejazz/mpc-trust-cbf/blob/main/media/simple_main/final_vid.gif)

The camera simulated is shown in picture-in-picture mode in the animation above. One can notice the deflection of the car's path towards the left owing to the less trustworthy pedestrian detected on the right. This is also plotted as follows:

![CARLA_main_plot](https://github.com/saadejazz/mpc-trust-cbf/blob/main/media/simple_main/plot.png)

## Simplified Formulation

A simplified formulation (single iterator as model) of the control algorithm is presented in the notebook ```mpc-trust-cbf.ipynb```. On a basic level, the higher the trust, the more aggressive the movement of the robot/vehicle. There is of course a ceiling to this which is the safety margin. The plot below shows how different trust values affect the path of the ego agent. 

![simple_trust](https://github.com/saadejazz/mpc-trust-cbf/blob/main/media/trust_simple.jpg)

A lower trust will make the path farther from the safety margin. Itcan also be inferred as increasing the safety margin, though that is not exactly what happens because it is relevant to other obstacles, if present. This can be seen in the following animation of moving towards a goal with two obstacles in the way:

![simple_trust_anim](https://github.com/saadejazz/mpc-trust-cbf/blob/main/media/anims/notebook.gif)

## Moving Pedestrians - CARLA simulations

This algorithm also works with moving obstacles where the speed of the obstacles is considered to be known (it can be perceived easily in modern systems). The effect of difference in trust on the path of the ego agent can be seen in the following two simulation clips.

High Trust:  

![moving_high](https://github.com/saadejazz/mpc-trust-cbf/blob/main/media/moving_decent_trust/final_vid.gif)

Low Trust:

![moving_low](https://github.com/saadejazz/mpc-trust-cbf/blob/main/media/moving_low_trust/final_vid.gif)

The difference in path can be more easily noted in the comparison plots shown below, where the animated plot on the left is for the first case where the pedestrian has high trust, while the one on the right is for the second case where the pedestrian has low trust:  

![together](https://github.com/saadejazz/mpc-trust-cbf/blob/main/media/anims/together.gif)

## Requirements to run
While the simplified system detailed in ```mpc-trust-cbf.ipynb``` can be easily ran after installing ```scipy```, ```numpy```, and ```matplotlib```, the CARLA simulator needs the latest version of CARLA installed and running, along with the requirements to run the trust estimator [trusty](https://github.com/saadejazz/trusty). The simulation code for the first scenario in this README is present in ```simulate.py```, however, the client runs slowly owing to the non-linear optimization problem to solve along with the perception (which is done inside the simulator by using a camera attached to the ego vehicle). In most cases, the simulation would need to be replayed to see in close to real-time (see ```start_replaying.py``` details on that).

## Problems/Future Work
Pedestrians arriving unexpectedly or suddenly in the environment can cause the control problem to be infeasible. This is common when pedestrians are moving, even so if either the car or the pedestrian is moving at a high speed. A more robust approach needs to be worked on to deal with this. The momentum feature of the trust estimator introduces smooth transition of trust which introduces some robustness.



