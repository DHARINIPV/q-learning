# Q Learning Algorithm


## AIM
To develop and evaluate the Q-learning algorithm's performance in navigating the environment and achieving the desired goal.

## PROBLEM STATEMENT

The goal is to implement a Q-learning algorithm for training a reinforcement learning agent in a given environment. The agent should use an epsilon-greedy strategy for action selection, balancing exploration and exploitation. Implement decay schedules for the learning rate (alpha) and exploration rate (epsilon) over episodes. Track the evolution of the Q-values and the learned policy across episodes. Evaluate the agent's performance across multiple random seeds, computing the average Q-values, value function, and policy to assess the stability and effectiveness of the learning process.

## Q LEARNING ALGORITHM
### Step 1:
Initialize the Q-table with zeros for all state-action pairs based on the environment's observation and action space.
### Step 2:
Define the action selection method using an epsilon-greedy strategy to balance exploration and exploitation.
### Step 3:
Create decay schedules for the learning rate (alpha) and epsilon to progressively reduce their values over episodes.
### Step 4:
Loop through a specified number of episodes, resetting the environment at the start of each episode.
### Step 5:
Within each episode, continue until the episode is done, selecting actions based on the current state and the epsilon value.
### Step 6:
Execute the chosen action to obtain the next state and reward, and compute the temporal difference (TD) target.
### Step 7:
Update the Q-value for the current state-action pair using the TD error and the learning rate for that episode.
### Step 8:
Track the Q-values, value function, and policy after each episode for analysis and evaluation.

## Q LEARNING FUNCTION
### Name: Dharini PV
### Register Number: 212222240024

```python
from tqdm import tqdm_notebook as tqdm
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))
    alphas  = decay_schedule ( init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    epsilons = decay_schedule(init_epsilon, 
                              min_epsilon, 
                              epsilon_decay_ratio, 
                              n_episodes)
    for e in tqdm(range(n_episodes), leave=False): # using tqdm
      state, done = env.reset(), False
      while not done:
        action = select_action(state, Q, epsilons[e])
        next_state, reward, done,_=env.step(action)
        td_target = reward + gamma * Q[next_state].max() * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state = next_state
      Q_track[e] = Q
      pi_track.append(np.argmax(Q, axis=1))
    V=np.max(Q, axis=1)
    pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]     

    # Write your code here
    
    return Q, V, pi, Q_track, pi_track

from tqdm import tqdm_notebook as tqdm
Q_qls, V_qls, Q_track_qls = [], [], []
for seed in tqdm(SEEDS, desc='All seeds', leave=True):
    random.seed(seed); np.random.seed(seed) ; env.seed(seed)
    Q_ql, V_ql, pi_ql, Q_track_ql, pi_track_ql = q_learning(env, gamma=gamma, n_episodes=n_episodes)
    Q_qls.append(Q_ql) ; V_qls.append(V_ql) ; Q_track_qls.append(Q_track_ql)
Q_ql = np.mean(Q_qls, axis=0)
V_ql = np.mean(V_qls, axis=0)
Q_track_ql = np.mean(Q_track_qls, axis=0)
del Q_qls ; del V_qls ; del Q_track_qls
```
## OUTPUT:

![image](https://github.com/user-attachments/assets/60ae8622-6520-4487-9ef7-d20075d460fa)

![image](https://github.com/user-attachments/assets/20ce43c7-9829-4553-9a7e-81c07f9898e9)

![image](https://github.com/user-attachments/assets/91dc8922-7b17-428d-b058-e896770fc646)

## RESULT:

Thus to develop and evaluate the Q-learning algorithm's performance is executed successfully.
