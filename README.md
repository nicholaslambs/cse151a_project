# CSE151A Project
Our project theme is revolved around training a reinforcement learning (RL) model on winning the card game, Uno. Since our project is utilizing reinforcement learning, our data cannot be directly grabbed from online resources and has to be generated through simulating gameplay. We contacted the Professor to recommend any suggestions to how to answer the MS2 and MS3 questions because of this challenge and suggested we think of our data and implementation at a high level, thus leading to the rest of this README.

You can find more details on the milestone READMEs in the `milestones` folder or by clicking [here](https://github.com/nicholaslambs/cse151a_project/blob/main/milestones/ms5/README.md) for the most recent milestone README.

We aim to optimize this model by tweaking the reward system of the model so that it can obtain the best winning choices (i.e. distinguishing between good and poor moves given the current game state).

Currently, our simulation comes from using the RL Card library to train our model. You can find their documentation about Uno [here](https://rlcard.org/games.html#uno). The RL Card library provides a strong framework for the purposes of our project where we have ready access to the simulated Uno environment and a variety of tools to train a reinforcement learning model.

For additional references to MS2 answers, please refer to [this other Markdown file](https://github.com/nicholaslambs/cse151a_project/blob/main/milestones/ms2/README.md)

For additional references to MS3 answers, please refer to [this other Markdown file](https://github.com/nicholaslambs/cse151a_project/blob/main/milestones/ms3/README.md)

For additional references to MS4 answers, please refer to [this other Markdown file](https://github.com/nicholaslambs/cse151a_project/blob/main/milestones/ms4/README.md)

For additional references to MS5 answers, please refer to [this other Markdown file](https://github.com/nicholaslambs/cse151a_project/blob/main/milestones/ms5/README.md)

Our group is gathering and simulating all this within the Python Notebook, `rl_card.ipynb`. You can find the notebook in the `rl_card` directory or [here](https://github.com/nicholaslambs/cse151a_project/blob/main/rl_card/rl_card.ipynb)

## Self Implementation

One of the group mates, Jaehoon, has also worked very hard on implementing the simulated Uno environment and RL model from scratch. You can find his work in the `self_implementation` directory. The `self_implementation` directory contains the following files:
- [unorefactored.ipynb](./self_implementation/unorefactored.ipynb)
- [unosharedbase2.ipynb](./self_implementation/unosharedbase2.ipynb)

The `unosharedbase2.ipynb` file contains the initial implementation of the Uno environment and the RL model. The `unorefactored.ipynb` file contains the refactored implementation of the Uno environment and the RL model. The refactored implementation is more modular and easier to understand. You can also find the analysis of the progress and the results [here](./self_implementation/analysis.pdf).

Please take a look at the hard work Jaehoon has put into this project! He has done a great job in implementing the Uno environment and the RL model from scratch.

# The Rules of Uno
You can find the general rules of Uno, [here](https://en.wikipedia.org/wiki/Uno_(card_game)). Since our current implementation (as of MS2 submission) does not support most wild cards, we can ignore any rules regarding following wild cards.

To list a few general (but important) rules of Uno:
- a player must play a card to relieve their turn
- a player must draw cards until they're able to play a card
- if a player is able to play a card, they cannot draw a card
- if a player places a wild card (when properly implemented), the following player must follow the card by drawing the cards or stacking the wild card
- a card can only be placed over the top card if it matches the top card's color and/or number or if they place a wild card

## Game Representation
The way that the RL Card library represents the states of Uno is very interesting. The game state or the observation space in Uno is represented through a multi-dimensional array, detailing the current player's hand, the discard pile's top card (target), and a summary of what's known about the other players' hands. This state encapsulation allows an RL model to understand the current game situation, make decisions, and learn from the outcomes.

**Deck and Cards:** Uno is played with a special deck containing four colors (red, green, blue, yellow) of number cards (0-9) and action cards (Skip, Reverse, Draw Two, Wild, Wild Draw Four). RLCard encodes these cards and their functionalities within the game logic, ensuring that the deck initialization, shuffling, and dealing processes mirror the real Uno game.

## State Representation
**Observation Space:** The game state or the observation space in Uno is represented through a multi-dimensional array or a more structured format, detailing the current player's hand, the discard pile's top card (target), and a summary of what's known about the other players' hands. This state encapsulation allows an RL model to understand the current game situation, make decisions, and learn from the outcomes.

**Feature Encoding:** The state is encoded into feature planes to facilitate learning by the RL algorithms. These planes include the player’s hand, the current target card on the discard pile, and an approximation of the other players' hands, structured in a way that an RL algorithm can process (e.g., a 4x15 matrix for color and card type representation).

## Action Representation
**Action Space:** RLCard defines a discrete action space for Uno, with actions corresponding to playing one of the cards in the player's hand or drawing a card from the deck. Each possible action (e.g., playing a red 3, a green skip, drawing a card) is assigned a unique identifier within a predefined range.

**Legal Actions:** At any given state, the environment identifies and provides a subset of legal actions based on the current player's hand and the target card on the discard pile. This mechanism ensures that the RL algorithms operate within the game rules, selecting only feasible moves.

# Implementing the Game Agents
## Random Agent
The random agent is a simpler agent that does not learn from interactions with the environment. Instead, it makes decisions entirely at random, providing a baseline level of performance against which the learning agents can be compared.

At each step, the random agent selects an action randomly from the set of legal actions provided by the environment. There's no consideration for the state of the game or the potential outcomes of these actions.

## Reinforcement Learning DQN Agent
The DQN agent in RLCard is a more sophisticated type of agent that utilizes deep learning to make decisions. The DQN algorithm aims to approximate the optimal action-value function (Q-function), which gives the expected return of taking an action in a given state and following a certain policy thereafter.

Experience Replay: DQN uses an experience replay mechanism to break the correlations between consecutive steps by storing the agent's experiences at each time step in a replay buffer. A mini-batch of these experiences is randomly sampled and used to train the network, improving stability and efficiency.

Target Network: To further stabilize training, DQN employs a separate target network to generate the Q-value targets for updates. This target network has the same architecture as the primary network but its weights are updated less frequently to provide consistent targets.

Epsilon-Greedy Strategy: To balance exploration and exploitation, DQN typically uses an epsilon-greedy strategy, where the agent selects random actions with probability epsilon and the best-known action with probability 1-epsilon. Epsilon is often decayed over time to shift from exploration to exploitation.

Training and Update Rule: The training process involves using the Bellman equation to update the Q-values towards better estimates. The loss function usually used is the mean squared error between the predicted Q-values and the target Q-values computed using the reward observed from the environment and the Q-values from the target network.

---
**Resources and other credits**:
- https://github.com/bernhard-pfann/uno-card-game-rl
- https://web.stanford.edu/class/aa228/reports/2020/final79.pdf
- https://rlcard.org/

*Note:* The repository referenced above is not a repository we're referencing, but we were told to include this reference to avoid any AI violations (and other sorts of trouble).
