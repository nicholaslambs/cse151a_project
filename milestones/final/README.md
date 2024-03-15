# Final Submission

![image](https://www.godisageek.com/wp-content/uploads/Uno-review1.jpg)

## Introduction
The card game Uno, while appearing simplistic in nature, instead offers a rich oppurtunity for strategic gameplay and quick decision making. This project aims to understand these aspects by developing a Reinforcement Learning (RL) agent capable of mastering Uno. Unlike more deterministic games, Uno's unpredictable nature, retrieved from its draw mechanics and the variety of action cards, requires an RL agent to develop strategies that can adapt to rapidly changing game states and opponent behaviors. This task is not just a challenge; it's a feature that makes the task compelling. The project's coolness factor lies in the endeavor to teach a machine how to navigate a game that combines luck, strategy, and psychology, reflecting the multifaceted decision-making humans engage in daily.

The goal to model and predict gameplay in Uno opens up insights for studying human behavior in competitive collaborative situations. Games serve as simplified models of larger social and strategic systems, where individuals' choices impact each other in multiple ways. In the scope of Uno, we can observe and analyze patterns of behavior, risk-taking, bluffing, and adaptability—skills that are crucial in various aspects of human interaction and decision-making. This deeper understanding can inform the development of more nuanced AI systems capable of interacting with humans in meaningful ways, from negotiating and conflict resolution to enhancing collaborative problem-solving. Thus, the project not only pushes the envelope in game-playing AI but also contributes to our knowledge of human cognitive and social behaviors, offering valuable lessons that can be applied both within and beyond the gaming context.

## Methods
For our RL agent, we decided to use a Q-Learning approach. This method focuses on learning the value of taking a specific action in a given state, aiming to maximize the total reward over time. This methodology is particularly suited to Uno, where the game's state changes dynamically with each turn, and the agent must decide from a finite set of possible actions—ranging from playing a card to drawing from the deck—based on the current state of the game.

Our method involves training a deep neural network to estimate the expected rewards for taking each possible action within a given state. Initially, the network's predictions are based on randomly initialized weights, reflecting the agent's initial lack of understanding the game's dynamics. As the agent plays more games and receives feedback in the form of rewards, we use this data to update the network's weights, improving its predictions about the game's outcome. This process is facilitated by the DQN algorithm's ability to balance exploration of new strategies with the exploitation of known actions (through the use of a Monte Carlo Tree Search). Our state representation for the neural network includes detailed features of the game, such as the agent's current hand, the top card of the discard pile and opponents' hand size providing a comprehensive view of the game's context.

In particular, we used RL Card library to obtain and simulate the Uno environment, as well as training our DQN (Deep Q Network). You can find their documentation about Uno [here](https://rlcard.org/games.html#uno). 

### Data Exploration
Because of the nature of our task, our data would not be accessible through online resources or already gathered data. For this reason, we had to gather retrieve our own data by running through thousands of simulation, and feeding the agent with the data from playing against other agents. Thus, we pitted 3 'random' agents against each other to observe how many turns a typical random agent would take to win the game, and retrieved the following results:

![image](https://github.com/nicholaslambs/cse151a_project/assets/57384225/401eb7f6-c681-471b-9a7d-caa4d5ff385a)

From this data, we can see that most of these Uno games end in around 25-50 games (when playing against a relatively small number of players). For this reason, we can observe that the RL agent can use this information to learn several things:

- adjust the reward function by winning in fewer turns
- understanding the common trend of where a player might win
- judging the RL agent based on the distribution of turns (i.e reward the agent more if they win really early, and less if it takes longer).

Similarly, we can also observe the expected number of action types that a typical Uno agent would do, as seen in the following bar chart:
![image](https://github.com/nicholaslambs/cse151a_project/assets/57384225/27a96976-aac5-4c93-b412-ea00af98d205)

It can be seen that drawing cards has the highest action count for random agents, so we instinctly knew to penalize the card draw action. Furthermore, we expect our RL agent to have a better distribution with respect to how many cards it draws - especially when it doesn't need to!

### Preprocessing
As mentioned earlier, the data that we're using is the Uno game environment itself. Because of this, we did not have to clean the data, but we did have to filter it to make it more manageable. Thus, we filted the data to only include the necessary information for the model to make decisions. Thanks to the RL Card library, the environment was structured in a way that is easy to work with. The RL Card library represents the states of the game in a multi-dimensional matrix (i.e. a 4x15 matrix for color and card type representation). More specifically:
- the current player's hand
- the discard pile's top card (target)
- the cards that have been played in the game
- a summary of what's known about the other players' hands

The states are encoded into seven feature planes (each plane is 4x15). This following details are provided from the RL Card library:
> The size of each plane is 4*15. Row number 4 means four colors. 
> Column number 15 means 10 number cards from 0 to 9 and 5 special cards—“Wild”, “Wild Draw Four”, “Skip”, “Draw Two”, and “Reverse”.
> Each entry of a plane can be either 1 or 0. 

In terms of the action space defined by the RLCard environment, each action is assigned an integer, representing the following:
 - 0-9, 10-12: Red number cards, Red action cards
 - 13: Red wild card
 - 14: Red Draw 4
 - 15-24, 25-27: Green number cards, Green action cards
 - 28: Green wild card
 - 29: Green draw 4
 - 30-39, 40-42: Blue number cards, Blue action cards
 - 43 Blue wild card
 - 44 Blue draw 4
 - 45-54, 55-57: Yellow number cards, Yellow action cards
 - 58: Yellow wild card
 - 59: Yellow draw 4
 - 60: Draw a card
   
The processing of this data (i.e. the state representation) is done by the RL Card library, so we did not have to do any additional processing. Our job is primarily to adjust the reward system to better train the model from the default system (i.e. +1 for winning, -1 for losing).

### Model 1
For our first model, we used a DQN with a multilayer perceptron consisting of two layers containing 64 hidden units, a learning rate of 0.0005 and a batch size of 32. This is reflected in our model initialization:

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

Furthermore, we decided to use a simple reward function as follows:
```python
def adjust_rewards(trajectories, payoffs):
    adjusted_trajectories = []
    for traj in trajectories:
        adjusted_traj = []
        for state, action, reward, next_state, done in traj:
            if action == 60:  # draw a card
                reward -= 1  # Penalty for drawing a card
                
            # Add more conditions for other strategic rewards
            # increment reward for actions 0-9, play red cards
            elif 0 <= action <= 9:
                reward += 1
            # increment reward for actions 15-24, play green cards
            elif 15 <= action <= 24:
                reward += 1
            # increment reward for actions 30-39, play blue cards
            elif 30 <= action <= 39:
                reward += 1
            # increment reward for actions 45-54, play yellow cards
            elif 45 <= action <= 54:
                reward += 1
                        
            adjusted_traj.append((state, action, reward, next_state, done))
        adjusted_trajectories.append(adjusted_traj)
    return adjusted_trajectories
```
As seen above, this simplistic reward function rewards the agent only when it does a good legal action (i.e play any card, as long as it does not draw a card). 
Finally, our training loop consisted of playing the RL agent against a random agent in the Uno environment, adjusting the trajectories of our model with respect to the reward fuction. This is seen as follows: 
```python3
episode_num = 25000  # Number of episodes 

evaluate_every = 1000 # Evaluate the agent every X episodes
evaluate_num = 100  # Number of games played in evaluation

with Logger(log_dir) as logger:
    for episode in tqdm(range(episode_num)):  # Number of episodes

        trajectories, payoffs = env.run(is_training=True)

        # Assuming 'payoffs' are the game outcomes for each player
        for i, payoff in enumerate(payoffs):
            if payoff > 0:  # Assuming a positive payoff means winning
                payoffs[i] = 100
            else:
                payoffs[i] = -25

        trajectories = reorganize(trajectories, payoffs)

        # After reorganizing the trajectories, adjust the rewards
        trajectories = adjust_rewards(trajectories, payoffs)
        # print(trajectories[0])

        for ts in trajectories[0]:
            block()
            agent.feed(ts)
            unblock()
        
        if episode % evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        evaluate_num,
                    )[0]
                )
```

This training loop remained consistent throughout all 3 models trained. 
The notebook containing this first implementation can be found HERE TO DO PLEASE ADD REFERENCE TO A NOTEBOOK

### Model 2
For our second model, we wanted to focus on improving our reward system and see its impact without altering the architecture of the agent too much. So we used the same model as in __Model 1__, but changed the reward function as follows:

```python
for state, action, reward, next_state, done in traj:
    if action == 60:  # Draw a card
        reward -= 1  # Penalty for drawing a card

    elif action >= 0 and action <= 9:  # Red number cards
        reward += 1
    elif action >= 10 and action <= 12:  # Red action cards
        reward += 3
    elif action == 13:  # Red wild card
        reward += 6
    elif action == 14:  # Red wild and draw 4 card
        reward += 10

    elif action >= 15 and action <= 24:  # Green number cards
        reward += 1
    elif action >= 25 and action <= 27:  # Green action cards
        reward += 3
    elif action == 28:  # Green wild card
        reward += 6
    elif action == 29:  # Green wild and draw 4 card
        reward += 10

    elif action >= 30 and action <= 39:  # Blue number cards
        reward += 1
    elif action >= 40 and action <= 42:  # Blue action cards
        reward += 3
    elif action == 43:  # Blue wild card
        reward += 6
    elif action == 44:  # Blue wild and draw 4 card
        reward += 10

    elif action >= 45 and action <= 54:  # Yellow number cards
        reward += 1
    elif action >= 55 and action <= 57:  # Yellow action cards
        reward += 3
    elif action == 58:  # Yellow wild card
        reward += 6
    elif action == 59:  # Yellow wild and draw 4 card
        reward += 10
```

As you can see, we adjusted the reward system to give the agent more rewards for playing special action cards. More specifically, the reward system was adjusted as follows:
- for playing a normal card move, small positive reward (+1)
- for drawing a card, small negative reward (-1)
- for playing a special action card, a larger positive reward (ranging from +3 to +10)
- winning the game, a large positive reward (+100)
- losing the game, a large negative reward (-25)

Furthermore, we added some complexity to the model by looking beyond playing certain types of cards and focusing on the state of the game (i.e the number of cards the opponet has) as seen below:

```python
# Actual game state details
raw_obs = state['raw_obs']

# Retrieve the number of cards in player's hand
num_cards_player = len(raw_obs['hand'])

# Provide the number of cards for each player with the current player being index 0
num_cards_opponent = raw_obs['num_cards'][1] if raw_obs['current_player'] == 0 else raw_obs['num_cards'][0]

if action == 60:  # Draw a card
    reward -= max(1, 3 - num_cards_player / 7)

# Adjust rewards for action cards based on the opponent's hand size
action_card_reward_multiplier = max(1, (7 - num_cards_opponent) / 7)

if action in range(10, 15) or action in range(25, 30) or action in range(40, 45) or action in range(55, 60):
    reward += 2 * action_card_reward_multiplier

if action in range(0, 10) or action in range(15, 25) or action in range(30, 40) or action in range(45, 55):
    reward += 1 + (3 - num_cards_player / 7)
```

As you can see, we adjusted the reward system to give the agent a dynamic set of rewards, depending on the state of the game. More specifically, the reward system was adjusted as follows:
- for playing a normal card move, small positive reward (+1) with a dynamic reward based on the number of cards in the agent's hand
- for drawing a card, small negative reward (-1) with a dynamic reward based on the number of cards in the agent's hand
- for playing a special action card, a larger positive reward (ranging from +2 to +10) with a dynamic reward based on the number of cards in the opponent's hand
- winning the game, a large positive reward (+100)
- losing the game, a large negative reward (-25)

The notebook containing these changes can be found HERE TODO ADD REFERENCE TO NOTEBOOK PLEASE

### Model 3
In this final model implementation, we varied the DQN architecture slightly by changing the number of hidden units in the multilayer perceptron to 128, as 001well as a larger learning rate of 0.001. Thus, our new DQN agent was initialized as follows:
```python 
agent = DQNAgent(
                 num_actions=env.num_actions,
                 state_shape=env.state_shape[0],
                 mlp_layers=[128,128], # Changed from 64
                 replay_memory_size=5000,
                 update_target_estimator_every=100,
                 epsilon_decay_steps=10000,
                 learning_rate=0.001, # Changed from 0.0005
                 batch_size=32,
                 device=get_device(),
                 save_path=log_dir
                 )
```

Furthermore, we adjusted the reward function slightly to incorporate a winning factor by incentivizing the agent to accelerate reaching the end of the game (i.e increase rewards that can conclusively lead to winning). This is seen here:

```python3
def adjust_rewards(trajectories, payoffs):
    adjusted_trajectories = []
    for traj in trajectories:
        adjusted_traj = []
        for state, action, reward, next_state, done in traj:
            # Actual game state details
            raw_obs = state['raw_obs']
            
            # Retrieve the number of cards in player's hand
            num_cards_player = len(raw_obs['hand'])
            
            # Provide the number of cards for each player with the current player being index 0
            num_cards_opponent = raw_obs['num_cards'][1] if raw_obs['current_player'] == 0 else raw_obs['num_cards'][0]
            
            # A larger reward if the agent has fewer cards
            winning_factor = (7 - num_cards_player) / 7

            if action == 60:  # Draw a card
                reward -= max(1, 3 - num_cards_player / 7)

            # Adjust rewards for action cards based on the opponent's hand size
            action_card_reward_multiplier = max(1, (7 - num_cards_opponent) / 7)

            if action in range(10, 15) or action in range(25, 30) or action in range(40, 45) or action in range(55, 60):
                reward += 2 * action_card_reward_multiplier * winning_factor

            if action in range(0, 10) or action in range(15, 25) or action in range(30, 40) or action in range(45, 55):
                reward += 1 + (3 - num_cards_player / 7) * winning_factor
                
            if num_cards_player <=2:
                reward +=5

            adjusted_traj.append((state, action, reward, next_state, done))
        adjusted_trajectories.append(adjusted_traj)
    return adjusted_trajectories
```

## Results
This will include the results from the methods listed above (C). You will have figures here about your results as well.
No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

## Discussion
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

## Conclusion
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts

## Collaboration
This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!

Name: Title: Contrtibution If the person contributed nothing then just put in writing: Did not participate in the project.
