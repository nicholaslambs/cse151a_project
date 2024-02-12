# CSE151A Project
Our project theme is revolved around training a reinforcement learning (RL) model on winning the card game, Uno. 

We aim to optimize this model by tweaking the reward system of the model so that it can obtain the best winning choices (i.e. distinguishing between good and poor moves given the current game state).

Currently, our simulation does not support some cards from the game (+4 and wild cards) so there are some rules that are currently excluded but is planned to be added in the very near future. You can find our current implementation of the simulation and RL model in [this notebook](https://github.com/nicholaslambs/cse151a_project/blob/main/unosharedbase2.ipynb).

# The Rules of Uno
You can find the general rules of Uno, [here](https://en.wikipedia.org/wiki/Uno_(card_game)). Since our current implementation (as of MS2 submission) does not support most wild cards, we can ignore any rules regarding following wild cards.

To list a few general (but important) rules of Uno:
- a player must play a card to relieve their turn
- a player must draw cards until they're able to play a card
- if a player is able to play a card, they cannot draw a card
- if a player places a wild card (when properly implemented), the following player must follow the card by drawing the cards or stacking the wild card
- a card can only be placed over the top card if it matches the top card's color and/or number or if they place a wild card

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
- `{"2", 2}`: represents a green colored card that has a number label 2

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

For now, we are thinking of starting simple with having the agent make its decisions based off of the agent's current hand, the current top/face up card, and how big the penalty is. 

Additionally, we were thinking of tackling a potential problem with our agent's hand representation in two different ways:
- Encoding the hand as a vector of 52 points, 1 for each distinct card. This means for some cases, the vector might store 0, 1, or 2, depending on how many of a distinct card the agent is holding.
  - For example, if the agent has both (Red, 1) cards in hand, then it would be a 2 in the encoded vector array. However, this might lead to normalization issues later.
- Encoding the hand as a vector of 100 points, 1 for every single card. With this implementation the hand is essentially one-hot encoded, but the tradeoff is that it has twice as much data which could make training slower.

Encoding the current top card will be done in a one-hot array of size 17. The first 4 will be one-hot encoded for the color of the card, and the next 13 will be one-hot encoded for the value of the card.

The penalty is the number of cards the agent has to draw. This is potentially difficult to implement, since we have to encode it in a finite number of units but it could technically stack for a really long time (in the case where players keep stacking Plus2 cards). So we propose starting with an array of size 3, one-hot encoded for how high the penalty is. If it's 1 (for drawing one card, the standard) then the vector would be `[0, 0, 0]`. For every Plus2 card that's stacked, the vector has a 1 slightly further to the right. If there are 3 or more Plus2 cards stacked, then the vector would be `[0, 0, 1]`.
