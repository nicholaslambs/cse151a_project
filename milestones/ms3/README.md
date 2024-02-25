# Milestone 3

In the previous milestone, we previously attempted using our own environment and model. However, to make better use of our time and resources, we decided to use the RL Card library to train our model. You can find their documentation about Uno [here](https://rlcard.org/games.html#uno). The RL Card library provides a strong framework for the purposes of our project where we have ready access to the simulated Uno environment and a variety of tools to train a reinforcement learning model. 

Since we've moved on to using the RL Card library, we updated the project's README to reflect any changes in our project.

## Finish major preprocessing
The data that we're working with is the Uno game environment. We did not have to clean the data, but we did have to filter it to make it more manageable. We filtered the data to only include the necessary information for the model to make decisions. 
The RL Card library structures the data in a way that is easy to work with. The state is encoded into feature planes to facilitate learning by the RL algorithms. 

These planes include the player’s hand, the current target card on the discard pile, and an approximation of the other players' hands, structured in a way that an RL algorithm can process (e.g., a 4x15 matrix for color and card type representation). 

Using the RL Card library, we're able to easily extract any features and other necessary information that may be relevant to the model. For example, for our first model, we used the action space to (hopefully) teach the model good and bad moves (i.e. playing a card, drawing a card).

In the future, we're planning on extracting additional features to better teach and train the model. Some ideas that we're considering include the number of cards in the player's hand, the number of cards in the other players' hands, and the number of cards in the draw pile.

The data is generated from every completed game and restructured so that the RL Card library can process it (and also becomes more readable for us). After getting reorganized and processed, we can use the data and update any necessary features to match our intended reward system to train the model.

## Train your first model
Following the second milestone, we moved onto using the RL Card library to train our first model. You can find their documentation [here](https://rlcard.org/). 

We used the DQN (Deep Q Network) model to train our first model. We used the following parameters:
```python 
agent = DQNAgent(
                 num_actions=env.num_actions,
                 state_shape=env.state_shape[0],
                 mlp_layers=[64,64],
                 replay_memory_size=5000,
                 update_target_estimator_every=100,
                 epsilon_decay_steps=10000,
                 learning_rate=0.0005,
                 batch_size=32,
                 device=get_device(),
                 save_path=log_dir
                 )
```

Since we're unfamiliar with DQN, we had to do some research to better understand the model. ...

In terms of training the model, we trained the model for 100000 episodes and then evaluated the model on a simulated Uno environment.
We evaluated the agent every 1000 episodes for 100 evaluation games and logged the average reward from the tournament simulation.

The environment had 2 players, and the model was trained to play against a random agent. A random agent is an agent that randomly selects a move from the available legal moves.

### How the Agent Interacts with the Game
The agent is set up to interact with the game by making decisions based on the current game situation, making decisions, and learning from the outcomes. The agent is trained to play against a random agent, which is an agent that randomly selects a move from the available legal moves. This means that the agent is structured to play legal moves by default, while our job is to train it to play good moves.

### Our current reward system
We adjusted the reward system slightly from our previous milestone. More specifically: 
- for playing a normal card move, small positive reward (+1)
- for drawing a card, small negative reward (-1)
- winning the game, a large positive reward (+100)
- losing the game, a large negative reward (-25)

Since it is our first model, we wanted to keep the reward system simple to see how small changes may affect the model.

## Data Representation
### State Representation
The RL Card library represents the states of the game in a multi-dimensional matrix (i.e. a 4x15 matrix for color and card type representation). More specifically:
- the current player's hand
- the discard pile's top card (target)
- the cards that have been played in the game
- a summary of what's known about the other players' hands

The states are encoded into seven feature planes (each plane is 4x15). This following details are provided from the RL Card library:
> The size of each plane is 4*15. Row number 4 means four colors. 
> Column number 15 means 10 number cards from 0 to 9 and 5 special cards—“Wild”, “Wild Draw Four”, “Skip”, “Draw Two”, and “Reverse”.
> Each entry of a plane can be either 1 or 0. 

The processing of this data (i.e. the state representation) is done by the RL Card library, so we did not have to do any additional processing. Our job is primarily to adjust the reward system to better train the model from the default system (i.e. +1 for winning, -1 for losing).

## Evaluate your model compare training vs test error
The way that RL Card suggests to evaluate the model is by running tournament simulations which essentially evaluate the model at the current state of training. We evaluated the agent every 1000 episodes for 100 evaluation games and logged the average reward from the tournament simulation. We have two sets of training data: (1) on 10,000 episodes and (2) on 50,000 episodes.

![fig_10K.png](fig_10K.png)

![fig_50K.png](fig_50K.png)

## Where does your model fit in the fitting graph.
From our current state of our model, we think that the model is actually learning, but only a very small amount. From the 50K figure, we can see that the model is actually starting to slowly average out around 0.2 reward average. This is a good sign, but we are still not satisfied with the model's performance. We believe that the model can be improved by training it on more episodes and by adjusting the reward system.

## What are the next 2 models you are thinking of and why?
Since our project is revolved around training a RL model to play Uno, we plan on training a model with a more complex environment. 

We plan on changing up our following models by adjusting the reward system and trying to see any more immediate changes in the model. 

We are also thinking of training the model on a 4-player environment to be more realistic to the actual game. However, we are aware that this will end up making the model more complex and harder to train.

## Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?
Since this is our first model, we are not expecting it to be perfect. We are expecting to see a lot of room for improvement. We are also expecting to see a lot of room for improvement in our understanding of the model and the environment. We are hoping to learn a lot from this first model and use that knowledge to improve our future models.

## What are the next steps?
We learned that it takes a very long time to train a model with the current setup. For following milestones, we plan on starting much earlier to avoid any stress relating to time. Additionally, we believe it'll be more beneficial for the model to be trained on even more iterations (i.e. more than 100,000 episodes). We also plan on training the model on a more complex environment, such as a 4 player environment. We believe that training the model on a more complex environment will allow us to better understand the model and the environment.