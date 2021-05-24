<h1 align =  "center">Propulsive Rocket Landing Simulation Using Q - Learning Approach </h1>

## Introduction

AI advancements are currently accelerating (no pun intended). 
Every year, researchers generate ground-breaking ideas, and this amazing field is still in its infancy. 
Reinforcement learning is one of the most fascinating areas of artificial intelligence. 
RL has been responsible for some of the most impressive AI examples, such as OpenAI developing an agent that outperformed the best professional human players.

RL has been responsible for some of the most exciting AI examples, such as OpenAI developing an agent that defeated the top professional human players in the e-sport DOTA 2! It is a huge accomplishment for an AI to be able to generate and learn complex strategies in a game as complex as DOTA 2, and it will push research even closer to achieving artificial general intelligence (AGI).

Aside from outperforming humans in games, RL can be used in a variety of fields such as financial trading, natural language processing, healthcare, manufacturing, and education.

In this project, we try to solve the problem of excessive cost of researching the optimal ways of landing a rocket with minimal damage to the rocket and minimal shocks to 
the rocket crew using the Reinforcement learning.

The implementation of the project starting from the agent uses the Deep Neural Network (DNN) to approximate a function of Q*(s, a), which circumvents the limitingthe standard infinite state spaces of the Q - learning algorithm.

For the implementation of the DNN, we used TensorFlow. A small batch of observations from this list have been selected altered, and are then employed as inputs for training the weights of the DNN.
The initial setup has been modified to overcome this problem and two networks have been implemented, the first network is Q network which is constantly updated by the Q – target network.

Finally, the agent takes an ε - greedy to choose a random action with a probability ε on each step and the best action the network has learned with a probability from 1 - ε.

We implement the hyperparameters to find the solution.

The solution implemented is based on three particular hyperparameters that had to be carefully selected to achieve successful results: 

A learning rate (α), a discount factor (γ) of future rewards, and a decline rate (ε-decay) to ensure proper operation and exploration balance.
The three values for α were 0.01, 0.001 and 0.0001.
The results were examined to find how the learning rate affects the agent's convergence during training.

## WHY DO WE NEED THIS?

Working with such powerful machines and engines is not something that everybody can afford. 

As a result, maintaining total precision and reliability in structural design becomes the cornerstone of every company that works in this area or issue.

The currently accessible software for these purposes is all licensed-based, such as MATLAB, Solid Works, Unigraphics, AutoCAD, and so on, and requires some formal experience along with some prior knowledge regarding the subject for generating a simulation.

Our project is more for those who are truly curious about how these amazing feats can be accomplished using some basic concepts of Reinforcement learning.

This project necessitates very basic or no knowledge of rocket science; rather, it is a code-based project that will aid in the understanding the basics Reinforcement learning concepts with a real-world application.

As a result, curiosity is maintained while concept clarity is increased.

## PRELIMNARY
### LunarLander-v2 Module OpenAi - Gym
The landing pad is always located at coordinates (0,0). 
The first two numbers in the state vector are the coordinates.
The reward for moving from the top of the screen to the landing pad at zero speed is approximately 100.140 points. If the lander moves away from the landing pad, it forfeits the reward. 
The episode ends if the lander crashes or comes to a stop, with additional -100 or +100 points awarded. Each leg's ground contact is worth +10. Each frame spent firing the main engine costs -0.3 points.

```Solved is worth 200 points. ```

It is possible to land outside of the landing pad. 
Since the fuel is infinite, an agent can learn to fly and then land on its first try. 


There are four distinct actions available: 
- Do nothing
- Fire the left orientation engine
- Fire the main engine
- Fire the right orientation engine

### Q-learning 
Q-learning is a model-free learning algorithm that learns the value of an action in a given state. 
It does not require an environment model (hence the term **_"model-free"_**), and it can handle problems with stochastic transitions and rewards without requiring adaptations.

Q in Q-learning stands for quality i.e., how much weightage each reward generated must be given in order to produce the maximum benefit.
Q-learning finds an optimal policy for every final decision process of Markov (FMDP) in order to maximise the expected value of the total award over each and every successive step from the current status. 
In view of the time spent exploring and a partially altered policy, Q-learning can identify an optimal action-selection policy for any FMDP. "Q" refers to the algorithm calculating function - the rewards expected in a given state for an action.

#### Properties of Q-Learning –
- Model Free Algorithm – Q-Learning does not require any pre - trained model or Markov Decision Process. 
The agent explores the environment and then learn from it. 
It'smore of like hit and trail method.

-  Temporal Difference – Here, predictions are re-evaluated after taking a step. 
It does not wait for the final result i.e. incomplete process are also helpful in predicting final results.

- Off-Policy-Learning – It calculates the value for optimal action-value pair using greedy approach i.e., it always selects that value that appears most relevant at that instant. 

- Exploration vs Exploitation Trade-Off - By the statement exploration and exploitation trade-off means that whether the agent is more benefitted by using exploration or by using exploitation i.e., which selected method gives more reward. 
If the agent is more benefitted by selecting random actions it will use that and if the agent is generating more rewards using the previously generated action it will use that. 

<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119306032-68e7e200-bc1e-11eb-8ca0-758a8eca7a00.png'</p>
<h5 align = 'center'> Fig-1: Illustration of ε-greedy approach </h5>

### Q-Learning Algorithm 

- Initialize Q-table – Initially Q-learning table must be created such that number of columns should be equal to number of actions and number of rows be equal to number of states.
- Choose an action – The most suitable action among all must be selected which gives the best reward at that particular instant.
- Perform an action – Selected action must be performed. Here comes the role of exploration vs exploitation trade off.
- Measure Reward
- Update Q-table – new value obtained must be updated in the Q-table.

<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119306156-a0568e80-bc1e-11eb-9edb-4d1d59161410.png'</p>
<h5 align = 'center'>Fig-2: Illustration of Q-learning algorithm</h5>

### Adam Optimisation

Adam is an algorithm for optimisation that can be used to update network weight iteratively based on training data instead of the standard stochastic gradient descent procedure.

Adam Configuration Parameters are:
- **Alpha** Also referred to as the learning rate or step size. The proportion that weights are updated. Larger values result in faster initial learning before the rate is updated. Smaller values slow learning right down during training
- **Beta1** The exponential decay rate for the first moment estimates.
- **Beta2** The exponential decay rate for the second-moment estimates. This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).

- **Epsilon** Is a very small number to prevent any division by zero in the implementation. 

#### Properties of Adam are –
-  Adam's true step size is approximately limiting the step size hyper-parameter in every iteration. This property adds to the previous non - intuitive hyperparameter of the learning rate intuitive understanding.
- The stage of the Adam update rule is invariant to the gradient magnitude, which helps a lot in areas with small gradients (such as saddle points or ravines). SGD is struggling in these areas to navigate quickly.
- Adam was designed to combine Adagrad's advantages with the sparse gradients, and RMSprop is working in good online environments. With these two, we can use Adam for a wider variety of tasks. The combination of RMSprop and SGD with momentum can also be considered for Adam. 


The Adam can be described as combination of two additional extensions of stochastic descent. In particular:
- **Adaptive Gradient Algorithm (AdaGrad)**, is an adaptive gradient algorithm that maintains a per parameter learning rate, which improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).

- **Root Mean Square Propagation (RMSProp)**, which also maintains per - parameter learning rates based on the average of recent magnitudes of gradients for the weights (e.g. how quickly it is changing). This implies that the algorithm performs well on both online and non-stationary problems (e.g. noisy).

### Equation of Q - learning

```New Q(𝑠,𝑎) = Q(𝑠,𝑎) + 𝛼 R(𝑠,𝑎) + 𝛾 max𝑄′(𝑠′,𝑎′) – Q(𝑠,𝑎)] …(1)```

Where,


> 𝜶: the learning rate, set between 0 and 1. Setting it to 0 means that the Q-values are never updated, hence nothing is learned. Setting a high value such as 0.9 means that learning can occur quickly.

> 𝜸: discount factor, also set between 0 and 1. This model the fact that future rewards are worth less than immediate rewards. Mathematically, the discount factor needs to be set less than 0 for the algorithm to converge.

> max𝑸′: the maximum reward that is attainable in the state following the current one. i.e the reward for taking the optimal action thereafter.

> NewQ: New Q value for that state and that action.

> Q(𝑠,𝑎): Current Q value.

> R(𝑠,𝑎): Reward for taking that action at that state.


## Implementation
Now we begin with our main part i.e, implementation part of the project. To give a high-level description of the entire process of as to how it is getting the output we desire.

<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119306247-c0864d80-bc1e-11eb-9f87-926312e8578e.png'</p>
<h5 align = 'center'>Fig-3: Basic block diagram of the process</h5>

### Implementing the agent

The Propulsive Rocket Lander has an infinite space across eight continuous dimensions, making it impossible to apply standard Q-learning, unless the space is discreet — inefficient and not practical to this problem. The DQN instead uses the Deep Neural Network (DNN) to approximate a function of Q*(s, a), which circumvents the limiting of the standard infinite state spaces Q-learning algorithm.

For the implementation of the DNN, we used TensorFlow. A small batch of observations from this list have been selected altered, and are then employed as inputs for training the weights of the DNN – a process called "Experience Replay." 
The agent's experience while acting in the environment was saved in a memoir buffer. 

The network uses an optimizer Adam and has a layer structure that has a node for each one of the state space's 8 dimensions, a hidden layer of 32 neurons and the output layer that maps each of the 4 possible lander actions.

The layers can be activated by a rectifier (ReLU) and no activation function is available for the output layer.Although this setup was very easy, the agent could not learn how to land consistently. The agent could not exceed very low rewards in most experiments, while others were able to achieve a very high level of rewards for a while, but diverged quickly from the management back to low rewards. 

The direct implementation of Q-learning with neural networks, like mentioned (Lillicrap et al., 2015), is unstable in many environments if it is also used in the computation of objective values by the same constantly updated network 

The initial setup has been modified to overcome this problem and two networks have been implemented; one network — known as the Q network which is in the implementation source code — is constantly updated while one third network — known as the Q-target network — has been used to predict the target value and updated after each episode from the weights of the Q-target network. This change stabilised the agent's performance, removing the constant divergence in many scenarios.Finally, the agent takes an ε - greedy to choose a random action with a probability 'ε' on each step and the best action the network has learned with a probability '1 - ε'. 

After each episode, the ε value is constantly decreased, which allows the agent first to explore all the possibilities and then to exploit optimal policies in more depth.

### Hyperparameter search
The solution implemented is based on three particular hyperparameters that had to be carefully selected to achieve successful results:

- A learning rate (α)
- A discount factor (γ) of future rewards 
- A decline rate (ε-decay) to ensure proper operation and exploration balance. 

The solution is used to achieve successfully.
A grid search for three possible values for every α and γ was performed in combination with ε - decay constant to determine the best values. 

A total of 5,000 iterations were used to train the operator so ε - decay would decline the exploration rates to approximately 0.05% by the end of training at 0.99941 percent (0.9941^5000 ~ 0.05) at the tuning of hyper parameters to 0.99941 during training.

This initial decline seemed reasonable to focus on α and γ so there was no reason to choose another initial value, as results were no better after testing different values. The three values for α were 0.01, 0.001 and 0.0001. The results were examined. If possible, values were more thoroughly searched, a slim-tuned rate would certainly have been achieved. However, the time limit did not allow this. These three values should allow us to properly examine how the learning rate affects the agent's convergence during training.

<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119306346-ddbb1c00-bc1e-11eb-8871-0fa95472db2d.png'</p>
<h5 align = 'center'>Fig-4: Learning rate and discount factor while epsilon decay = 0.99941</h5> 

The discount factor γ was explored at three values: 0.9, 0.99 and 0.999. The factor of discount indicates the agent's ability to achieve success in future actions. 
A small value such as ε = 0.9 prevents a successful credit of the agent after 10 steps in the future (1/ (1- γ) = 1/ (1-0.9) = 10) whereas the great value such as ε = 0.999 will enable the agent to look at a total of 1 000 actions for the future.

Given each episode of the deployed agent to be forcibly terminated has a limit of 1,000 steps, the grid search includes 0.9 for the evaluation of the myopic agent, 0.99 for the evaluation of a far-sighted agent and 0.99 for the moderate value between them. The above figure shows the accumulated reward for the last 100 episode of the agent and tries over 5000 episodes to match nine possible combinations of the rate and the reduction factor. (The highlights from the rest are in a blue colour, α = 0.0001 and ε = 0.99.)

Note how very poor results are with α = 0.01. A too large learning rate keeps the network constantly out of step and prevents convergence and sometimes even causes differences. The α = 0.001 learning rate shows better results, but only with an α = 0.0001 less convergence is achieved. The discount rate is lower. Combining this value with the right discount factor, the environment has succeeded in completing the 200 accumulated prize mark at approximately 3600 times.

In view of the discount factor, a value of 0.9 did not lead to successful results because the agent was unable to pay the rewards achieved at the correct landing - the agency couldn't solve the problem of temporary credit assignment for more than 10 steps ahead. A much higher discount factor of 0.999 did not push the agent quickly enough before an episode ended, while allowing the agent to properly obtain credit benefits by landing properly.The agent could fly indefinitely without any rush to land until the end of the episode. 

Finally, a moderate discount factor of 0.99 caused the agent to land to collect the loan at landing as quickly as possible — in the future no more than 100 steps were creditable. This value of the discount factor was the best results from the experiments.A survey of the various values of ε - decay was performed with the best combination of the learn rate and discount factor selected. Four additional values have been tested alongside the default value 0.99941 — which was 99.95% by operation and 0.05% at training ends. 

Four more combinations were tested:
- 0.99910 (99.99% - 0.01%) 
- 0.99954 (99.90% - 0.10%)
- 0.99973 (99.75% - 0.25%)
- 0.9998773 (99.50% - 0.50%)


The results are as follows –

<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119306539-2a065c00-bc1f-11eb-9a5d-9884f0088350.png'</p>
<h5 align = 'center'>Fig-5: The alpha decay with learning rate as 0.0001 and discount factor at 0.99</h5>

Notice that for higher values, the agent does not use what it learns often enough, so it explores and does not benefit from what it learns. It does not use the agent. The agent uses what he's learned quickly but gets stuck because he can't explore other ways to increase his reward. The best result was using the initial value of 0.99941 for the agent to unveil the whole state space, which provides the right balance of exploration versus exploitation.

Other parameters are needed to successfully operate the lander. These values were selected by a method of testing and error due to time constraints and a comprehensive exploration did not take place to correctly adjust them. 

The following were the parameters:
- The memory replay buffer is 250.000 in size to ensure that the agent has discarded very previous experiences and uses only the latest 250 – 500 episodes.
- The size of the DNN replay kit was 32, but the results were similar in 64. Values were not consistently converged with a 16 or 128 mini batch.
- In order to allow the agent to converge appropriately, 5,000 episodes were set.
- After each step, the Q network weights were updated, but performance was sacrificed.

## Agent Results
Here is a look at the individual reward obtained by our agent while training using the best set of hyperparameters selected from the previous experiments. The accumulated reward is also displayed for context:

<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119306621-473b2a80-bc1f-11eb-9c1e-d12796ea2f33.png'</p>
<h5 align = 'center'>Fig-6: Rewards per training episode</h5>

Notice that during the first three thousand hits the agent's behaviour is extremely irritated, but as it approaches the end it will become less intermittent, as it begins to exploit what it has learnt.

<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119306678-5d48eb00-bc1f-11eb-8f14-7ac1b09b461c.png'</p>
<h5 align = 'center'>Fig-7: Rewards per testing episode</h5>

Finally, the figure above shows the agent's results following the completion of training. In this case, the agent was operated three times and the reward for each episode was drawn. Note that there were no crashes in the 100 episodes for one round that indicated that the training process was successful. The average premium per run is 209, 213 and 216, as shown in the chart legend.

## Conclusion
The primary challenge in training a successful offer to solve the environment of Lunar Lander was the time constraint imposed by the planning of the project, exacerbated by the algorithm's slow training time to achieve good results. 

In this way hyperparameter cannot be explored in more detail to obtain better results in terms of convergence, and the following list of possible improvements and research areas can be: 
- How often we train the DNN can lead to significant improvements in speed. The algorithm trains it to improve accuracy at the expense of performance after every step of the time.
- Training is currently done with a fixed rate of learning, but it may lead to better results if declined over time.
- A more exhaustive grid search beyond the nine exploration values should take advantage of an initial learning rate and discount factors.
- The current algorithm uses a ε-greedy strategy, but other approaches may lead to better results.
- The use of a dynamic-sized replay buffer might show improvements as shown in (Liu et al., 2017)
- The use of a Double Q-learning Network (DDQN) instead of a simpler DQN may increase the accuracy of the agent.

In addition to much scope for improvement, the current implementation has successfully resolved the environment and 88showed the importance of a proper hyperparameter selection to make the agent converge. This project specifically focused on the influence of study rate, discount factor and decline and the impact on the agent.

## References
1. Young, The Apollo Lunar Samples: Collection Analysis and Results, Praxis, New York, NY, USA, 2017.
2. Lunar Lander Strategies: 11th Biennial ASCE Aerospace Division International Conference on Engineering, Science, Construction, and Operations in Challenging Environments
3. Sutton, R. & Barto, A. Reinforcement Learning: An Introduction (MIT Press, 1998)
4. Diuk, C., Cohen, A. & Littman, M. L. An object-oriented representation for efficient reinforcement learning. Proc. Int. Conf. Mach. Learn. 240–247 (2008)
5. Watkins, C. J. & Dayan, P. Q-learning. Mach. Learn. 8, 279–292 (1992)
6. T. Y. Park, J. C. Park, and H. U. Oh, “Evaluation of structural design methodologies for predicting mechanical reliability of solder joint of BGA and TSSOP under launch random vibration excitation,” International Journal of Fatigue, vol. 114, pp. 206–216, 2018.
7. OpenAi Gym Official Documentation

## Team Of
- [Abhishek Chaudhary](https://github.com/chaudhary312)
- [Khushi Agrawal](https://github.com/khushi-411)
- [Prabhat Kumar](https://github.com/prabhatk579)
