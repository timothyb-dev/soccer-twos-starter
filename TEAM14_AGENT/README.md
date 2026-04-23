# Team 14 Agent

**Agent name:** Team14_MultiagentSelfPlay

**Author (s):** Joseph Schwalbe (jschwalbe3@gatech.edu), Divyam Kumar (dkumar75@gatche.edu) and Timothy Bernard (tbernard8@gatech.edu)

## Description

An agent trained with PPO via multi-agent self-play using Ray RLLib. This version does...

Reward modification: This agent has a modified reward function. Here, the agent gets -1 for each goal allowed, +1 - (1/MaxSteps) for each goal scored. This part of the reward is from the environment. In addition to this, our agent gets two other awards. Firstly, the agent gets an award for being close to the ball. This equation is as follows
reward += RewardScalar / (||player_position - ball_position|| + offset).
This gives reward gives additional rewards for being closer to the ball, which encourages interaction between the ball and the agent. 
The second reward modification has to do with the velocity of the ball. Here, each team/player gets an award based on the speed and direction of the ball. For example, the blue team gets position reward for the ball going to the right and negative reward for the ball going to the left. The equation for this is as follows
reward += ball_velocity_x * RewardScalar for blue team and 
reward += ball_velocity_x * -RewardScalar for orange team.
This reward modification encourages the agent to move the ball toward the opposing teams side, which in turn will result in more goals obtained.