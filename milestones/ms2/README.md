# Milestone 2 Data

# Data Exploration
Currently, we understand that our data is not accessible through online resources or already gathered data. 
We plan on gathering this data on our own by running through thousands of simulations. Additionally, we plan on feeding the RL Agent with the data in order for it to learn.

We believe that we're going to have to gather a lot of different types of data. For example, in order for our RL Agent to have strong data for it to learn, it needs to know all of the game states to make educated moves that would put itself in a winning position (i.e. using which cards have already been played to make educated moves or probability that other players have wild cards or probabiltity of which color is most likely to chain, etc.)

The data we collect is going to change significantly as we continue to toy around with our simulation and RL Agent because we're going to continuously learn new things about what is important.
Furthermore, we'll learn types of data might be important for the RL Agent to know so that it can gain a higher chance of winning. 

This puts in a situation where we're currently missing a lot of data. While this might sound alarming, we've already started gathering data that we think might be useful for the RL Agent to know about. 

## Data Plots
![image](https://github.com/nicholaslambs/cse151a_project/assets/57384225/401eb7f6-c681-471b-9a7d-caa4d5ff385a)

For this data, we wanted to simulated 100 games with 3 basic agents to see the average length of a game considering basic moves. From this data, we can think the RL Agent can use this to learn several things:
- adjusting the reward function by winning in fewer turns
- understanding the common trend of where a player might win
- judging the RL Agent based on the distribution of turns (i.e. positively reward if they win really early, a lot less reward if they win later on?)

![image](https://github.com/nicholaslambs/cse151a_project/assets/57384225/27a96976-aac5-4c93-b412-ea00af98d205)

For this data, we wanted to simulated 1000 with 3 basic agents to see the common trend with the game actions. Using this data, the RL Agent can better understand the typical action distribution within Uno and can adjust the strategy based on this data. The agent can also be better rewarded for choosing actions that are more common in successful games. Furthermore, the agent can learn the value of specific moves by taking specific action, given a particular game state (i.e. which cards have been played and the current top card).

We're still very much working hard to gather more important data, but it is difficult given how we need to be more creative on our perspective with the model and simulations.
