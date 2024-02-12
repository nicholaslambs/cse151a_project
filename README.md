# CSE151A Project

Our project theme is revolved around training a reinforcement learning (RL) model on winning the card game, Uno. 

We aim to optimize this model by tweaking the reward system of the model so that it can obtain the best winning choices (i.e. distinguishing between good and poor moves given the current game state).

Currently, our simulation does not support some cards from the game (+4 and wild cards) so there are some rules that are currently excluded but is planned to be added in the very near future.

# The Rules of Uno
You can find the general rules of Uno, [here](https://en.wikipedia.org/wiki/Uno_(card_game)). Since our current implementation (as of MS2 submission) does not support most wild cards, we can ignore any rules regarding following wild cards.

To list a few general rules of Uno:

## Game Representation
The game is initialized by the following actions, before any players are able to make any moves:
1. populates deck
2. the deck is shuffled
3. seven (7) cards are dealt to each player
4. one (1) card is placed into the discard and makes it the top facing card

We can refer to the following objects of the game by:
- game master: keeps track of current game state
  - deck of cards (list of tuples): what players draw additional cards from
  - discard pile (list of tuples): in order that cards have been played
  - top card (single tuple): represents the current face-up card on top of the deck
  - penalty (int): this is how many cards a player draws if they decide to draw 
- agents: current "players" in the game (also keeps track of whose turn it is)

Each card is represented as tuple that carries the card's (1) card color (red, green, blue, yellow) and (2) number value (1-9).
- {"2", 2}: represents a green colored card that has a number label 2

After the game has been initialized, the current player to make a move has the following options:
- play a card that either has the (1) same color as the top card and/or (2) same number as the top card
- if the player cannot play any of their cards, draw a card from the deck of cards until they're able to make a valid move

Once the player has made a move, then the next player is moved onto and the same choices are applied to them. We continue this until all players have moved at least once in an iteration and the order of the players will always be the same for continuing iterations. 

# Implementing the RL Agent
## The Strategies
**Basic** Agent:
- this is a simple agent that has its hand of cards represented as a list or array data structure
- it will be our 'control' player, which linearly searches through its hand from left to right and selects the first playable card
- it always plays legal moves
- for example, if Plus2 cards are being stacked, then it will try to play another Plus2 card if it has one in hand before drawing cards

**Reinforcement Learning** (RL) Agent:
- this is the agent we are trying to implement
- at first, we would like it so that it doesn't know anything except that on its turn, it should make a move
- this can include illegal moves, so hopefully with some training it learns not to make moves that are illegal and immediately lose it the game

## The Rewards
Our current reward system is as follows:

If we represent the reward for playing some move as an integer, we were thinking of initially trying these reward values:
- for playing a legal move, small positive reward (+1)
- for drawing a card, small negative reward, possibly proportional to the number of cards drawn (-1)
- for playing a card and winning the game (getting rid of its last card), a large positive reward (+100)
- for playing an illegal move and losing the game, a large negative reward (-100)

Since we're still in the early stages of our project, this reward system is definitely tentative and will most likely change as we continue to work with our RL model. 

Our current reward system feels a little barebones, but we are aiming to increase its complexity as we continue working on this project. For example, we definitely want the model to learn complex and smart decisions based on the previous game states (i.e. exactly what cards have been played may impact which card(s) it wants to play). 

## Goals Moving Forward
With this project, we aim to optimize the RL model by essentially becoming the agent. In the current state of our simulation and model, we see:
- our current hand of cards
- the top card
- our score (in terms of rewards)
- what we've learned from previous experiences (i.e. how making certain actions impact our reward)
