# coding: utf-8
import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

register(id='SlipperyFrozenLake-v1',
    entry_point='gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': True})
register(id='FrozenLake-v1',
    entry_point='gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False})

"""   
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [0, nS - 1] and actions in [0, nA - 1], P[state][action] is a list of
		tuple of the form (p_trans, next_state, reward, terminal) where
			- p_trans: float
				the transition probability of transitioning from "state" to "next_state" with "action"
                P[s][a][ss_id][0]
			- next_state: int
				denotes the state we transition to (in range [0, nS - 1])
                P[s][a][ss_id][1]
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to	"next_state" with "action"
                P[s][a][ss_id][2]
			- terminal: bool
			  True when "next_state" is a terminal state (hole or goal), False otherwise
                P[s][a][ss_id][3]
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, eps=1e-3):
    """Evaluate the value function from a given policy.
    Parameters
    ----------
    P, nS, nA, gamma:
      defined at beginning of file
    policy: np.array[nS]
      The policy to evaluate. Maps states to actions.
    eps: float
      Terminate policy evaluation when
        max |value_function(s) - next_value_function(s)| < eps
    Returns
    -------
    value_function: np.ndarray[nS]
      The value function of the given policy, where value_function[s] is
      the value of state s
    """
    
    value_function = np.zeros(nS)

    next_value_function=np.zeros(nS)
    k=0
    while True:
       k=k+1
       value_function = next_value_function
       next_value_function=np.zeros(nS)    # this is where the bug is!!!!!! gonna initite again before new iterataion begins.
       for s in range(nS):
          a=policy[s]
          for trans in P[s][a]:
              p,ss,r,terminal=trans
              next_value_function[s]+=p*(r+gamma*value_function[ss]*(1-terminal))
       if (np.max(np.fabs(value_function-next_value_function)) < eps):
           break

    return value_function

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    
    '''
    """Given the value function from policy, improve the policy.
    Parameters
    ----------
    P, nS, nA, gamma:
      defined at beginning of file
    value_from_policy: np.ndarray
      The value calculated from evaluating the policy
    policy: np.array
      The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    '''
    new_policy = np.zeros(nS, dtype=int)

    for s in range(nS):
        Q_s=np.zeros(nA)
        for a in range(nA):
            for p,ss,r,terminal in P[s][a]:
              Q_s[a]+=p*(r+gamma*value_from_policy[ss]*(1-terminal))
        new_policy[s]=np.argmax(Q_s)
    return new_policy

def policy_iteration(P, nS, nA, gamma=0.9, eps=10e-3):
    value_function = np.zeros(nS)
    
    previous_value=np.zeros(nS) 

    improved_policy = np.zeros(nS, dtype=int)

    k=0
    while True:
        k=k+1
        previous_policy=improved_policy
        
        previous_value= policy_evaluation(P,nS,nA,previous_policy,gamma,eps)

        improved_policy = policy_improvement(P,nS,nA,previous_value,previous_policy,gamma)
        
        value_function=previous_value
        if (np.all(previous_policy==improved_policy)):
            break
    return value_function, improved_policy


def value_iteration(P, nS, nA, gamma=0.9, eps=1e-3):  
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    next_value_function=np.zeros(nS)
    Q_function=np.zeros((nS,nA))
    
    while True:
      value_function=next_value_function
      for s in range(nS):
          for a in range(nA):
             # compute Q[s][a]
             Q_function[s][a]=0
             for i in range(len(P[s][a])):
                p,ss,r,terminal=P[s][a][i]
                Q_function[s][a]=Q_function[s][a]+p*(r+gamma*value_function[ss]*(1-terminal))    #*(1-terminal)
      next_value_function=np.max(Q_function, axis=1)
      policy=np.argmax(Q_function,axis=1)
      if np.max(np.abs(next_value_function-value_function)) < eps:
         break
    return value_function, policy

def render_single(env, policy, max_steps=100):
    """This function does not need to be modified.
    Renders policy once on environment. Watch your agent play.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_reward = 0
    state, _ = env.reset()
    
    k=0
    while True:
        env.render()
        #time.sleep(0.25)
        '''
        time.sleep(0.001)
        '''
        k=k+1
        #print(f"loop iteration {k}")
        action = policy[state]
        state, reward, terminal, truncated, _ = env.step(action)
        episode_reward += reward
        if terminal or truncated:
            #print(f"will break after k={k} steps")
            break
    env.render()
    if not terminal:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action
# You may change the parameters in the functions below
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments
    # human render mode for the animation
    
    env = gym.make("FrozenLake-v1", render_mode="human")
    #env = gym.make("SlipperyFrozenLake-v1", render_mode="human")

    env = TimeLimit(env, max_episode_steps=100) #100
    P = env.P
    nS, nA = env.observation_space.n, env.action_space.n

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(P, nS, nA, gamma=0.9, eps=1e-3)
    #time.sleep(1)
    render_single(env, p_pi, max_steps=100)
    print(p_pi)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(P, nS, nA, gamma=0.9, eps=1e-3)
    render_single(env, p_vi, max_steps=100)
    print(p_vi)
