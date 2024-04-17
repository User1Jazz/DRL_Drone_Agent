# DRL_Drone_Agent
This repository contains ROS2 package with Deep Reinforcement Learning algorithms implemented for drone navigation and obstacle avoidance.

### The following algorithms were implemented:
1. Deep Q Network (DQN)
2. Double Deep Q Network (DDQN)
3. Soft Actor Critic (SAC)

## Features
- Automated training: agents are capable of preparing themselves for the next training episode once the current episode ends
- Experience replay buffer
- Analysis charts: at the end of the training session, the ROS node will export the training data such as reward charts and loss values
- Model saving after the training session: the network parameters are saved for future use after the training session

## Requirements
- This package has been specifically developed to work with the [Drone Simulator](https://github.com/User1Jazz/Drone-Sim) made in Unity Game Engine
- [Drone Sim ROS Messages](https://github.com/User1Jazz/drone_sim_messages)
