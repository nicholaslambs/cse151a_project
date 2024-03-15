# Final Submission

## Introduction
The card game Uno, while appearing simplistic in nature, instead offers a rich oppurtunity for strategic gameplay and quick decision making. This project aims to understand these aspects by developing a Reinforcement Learning (RL) agent capable of mastering Uno. Unlike more deterministic games, Uno's unpredictable nature, retrieved from its draw mechanics and the variety of action cards, requires an RL agent to develop strategies that can adapt to rapidly changing game states and opponent behaviors. This task is not just a challenge; it's a feature that makes the task compelling. The project's coolness factor lies in the endeavor to teach a machine how to navigate a game that combines luck, strategy, and psychology, reflecting the multifaceted decision-making humans engage in daily.

The goal to model and predict gameplay in Uno opens up insights for studying human behavior in competitive collaborative situations. Games serve as simplified models of larger social and strategic systems, where individuals' choices impact each other in multiple ways. In the scope of Uno, we can observe and analyze patterns of behavior, risk-taking, bluffing, and adaptability—skills that are crucial in various aspects of human interaction and decision-making. This deeper understanding can inform the development of more nuanced AI systems capable of interacting with humans in meaningful ways, from negotiating and conflict resolution to enhancing collaborative problem-solving. Thus, the project not only pushes the envelope in game-playing AI but also contributes to our knowledge of human cognitive and social behaviors, offering valuable lessons that can be applied both within and beyond the gaming context.

## Methods
this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
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

### Model 1
### Model 2
### Model3

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
