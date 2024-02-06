# DRL_Drone_Agent
This repository contains drone ROS2 package

# Drone Agent structure

## Hyperparameters
- __Learning rate__ - rate of optimisation for neural networks
- __Discount factor__ - to adjust agent's behaviour towards gaining the rewards and prevent infinite horizon rewards
- __Exploration probability__ - probability of selecting a random action from the set of actions
- __Exploration decrease__ - the value by which the exploration probability is decreased after taking the random action (to encourage exploration at the start of the training and slowly move towards the exploitation strategy)

## Runtime Parameters
- __Preivous state__
- __Previous action__
- __Previous reward__
- __Current state__
- __Current action__
- __Current reward__

## Network Inputs
- __Imu data__ (linear and angular acceleration)
- __Height data__ (single float value for ground proximity)
- __Target position__ (XYZ of the target)

## Network Outputs
- __NumPy array__ of actions (9 of them: Stop, Move(Forward,Backward,Left,Right, Up, Down), Yaw(Left,Right))

## Architecture
- Double Dueling Deep Q Network (D3QN)
######
    Explain the architecture here
    
    Training loop:
    ----------------------------------------------------------------
    Where should I implement the experience replay?
    while not done:
        # Step 1: Get observation
        current_observation = get_observation()

        # Step 2: Pick an action (epsilon-greedy policy)
        action = epsilon_greedy_policy(current_observation)

        # Step 3: Record observation and action
        record_observation_and_action(current_observation, action)

        # Step 4: Perform action
        next_observation, reward, done = perform_action(action)

        # Step 5: Update replay buffer
        update_replay_buffer(current_observation, action, reward, next_observation, done)

        # Step 6: Optimize the networks based on the DQN loss
        optimize_networks()

        # Move to the next time step
        current_observation = next_observation
    ----------------------------------------------------------------