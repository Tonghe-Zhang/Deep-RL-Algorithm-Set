import torch
import numpy as np
from collections import deque

def get_buffer(cfg, **args):
    assert type(cfg.nstep) == int and cfg.nstep > 0, 'nstep must be a positive integer'
    if not cfg.use_per:
        if cfg.nstep == 1:
            return ReplayBuffer(cfg.capacity, **args)
        else:
            return NStepReplayBuffer(cfg.capacity, cfg.nstep, cfg.gamma, **args)
    else:
        if cfg.nstep == 1:
            return PrioritizedReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, **args)
        else:
            return PrioritizedNStepReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, cfg.nstep, cfg.gamma, **args)


class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size, device, seed):
        self.device = device
        self.states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous()
        self.actions = torch.zeros(capacity, action_size, dtype=torch.float).contiguous()
        self.rewards = torch.zeros(capacity, dtype=torch.float).contiguous()
        self.next_states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous()
        self.dones = torch.zeros(capacity, dtype=torch.int).contiguous()
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.states[self.idx] = torch.as_tensor(state)
        self.actions[self.idx] = torch.as_tensor(action)
        self.rewards[self.idx] = torch.as_tensor(reward)
        self.next_states[self.idx] = torch.as_tensor(next_state)
        self.dones[self.idx] = torch.as_tensor(done)

        # update counters
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        # using np.random.default_rng().choice is faster https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        batch = (
            self.states[sample_idxs].to(self.device),
            self.actions[sample_idxs].to(self.device),
            self.rewards[sample_idxs].to(self.device),
            self.next_states[sample_idxs].to(self.device),
            self.dones[sample_idxs].to(self.device)
        )
        return batch

class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, action_size, device, seed):
        super().__init__(capacity, state_size, action_size, device, seed)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        ############################
        '''
        Get n-step state, action, reward and done for the transition,
        discard those rewards after done=True

        output:
            In N step setting, 
            state  <- s_{t}        
            action <- a_{t}
            reward_look_ahead <-  \sum_{k=0}^{n-1} \gamma^k R_{t+k+1}(1-done_{t+k+1}) 
                    //we sum up until done==True, truncate the remained rewards 
            done   <-whether there is a done flag during the N steps from s_{t} to s_{t+N-1}
            
            Bellman update:

            Y_t^n=\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_a Q(s_{t+n}, a;w-)   

            In the implementations of the DQNagent, self.gamma = cfg.gamma ** cfg.nstep

            so in codes, the Bellman update looks like:

            Y_t^n=\sum_{k=0}^{n-1} cfg.gamma^k R_{t+k+1} + cfg.nstep* \max_a Q(s_{t+n}, a;w-)  

            In the initialization of the NstepReplyBuffer, the input parameter gamma is cfg.gamma
            so we can safely use it self.gamma of NStepReplayBuffer to indicate the absolute definition.
        '''

        # retrieve s_t, a_t and examine all the done flags for the n-steps.
        state, action, _, done = self.n_step_buffer[0]

        # calculate accumulated future reward=\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} 
        
        reward_look_ahead = 0
        for k, (_,_,r,d) in enumerate(self.n_step_buffer):
            reward_look_ahead+=self.gamma**k*r
            if self.n_step_buffer[k][3]:  # if done at index 3
                done = True
                break
        #####################
        return state, action, reward_look_ahead, done

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, action_size, device, seed):
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, action_size, device, seed)

    def add(self, transition):
        self.priorities[self.idx] = self.max_priority
        super().add(transition)

    def sample(self, batch_size):
        # sample_idxs = self.tree.sample(batch_size)
        sample_idxs = self.rng.choice(self.capacity, batch_size, p=self.priorities / self.priorities.sum(), replace=True)
        # Get the importance sampling weights for the sampled batch using the prioity values
        # For stability reasons, we always normalize weights by max(w_i) so that they only scale the
        # update downwards, whenever importance sampling is used, all weights w_i were scaled so that max_i w_i = 1.
        
        ############################
        '''
        inputs:
            batch_size:int  (128 by default)
        Returns:
            weights: torch.Size([N])
            batch: 5-element tensor tuple, each element is of shape torch.Size([N])
            where N is the batch_size (128 by default)
        '''
        # randomly generalize a suite of indices that
        # indicates the chosen batch of trajectories. 
        # sample_idxs = self.tree.sample(batch_size)
        sample_idxs = self.rng.choice(
            self.capacity, batch_size, 
            p=self.priorities/self.priorities.sum(),
            replace=True)
        # Get the importance sampling weights for the sampled batch using the prioity values
        # For stability reasons, we always normalize weights by max(w_i) so that they only scale the
        # update downwards, whenever importance sampling is used, 
        # all weights w_i were scaled so that max_i w_i = 1.
        ############################
        # We only update the sampled transitions in the replay buffer. 
        # P is the importance sampling weights for the sampled batch only.
        # P(i)=\frac{p_i^\alpha}{\sum_{j sampled} p_j^\alpha}
        # w_i = (1/N 1/P(i))^{\beta}

        p=self.priorities[sample_idxs]

        P=torch.as_tensor(p/p.sum())
        
        #N is the actual number of samples in the main buffer.
        # Notice that we should not assign self.capacity to N, but we should use self.size to define N.
        N= self.size

        weights=torch.pow(1/(N*P),self.beta)

        # normalize weights:
        weights=weights/max(weights)
        ############################
        ############################

        batch = (
            self.states[sample_idxs].to(self.device, non_blocking=True),
            self.actions[sample_idxs].to(self.device, non_blocking=True),
            self.rewards[sample_idxs].to(self.device, non_blocking=True),
            self.next_states[sample_idxs].to(self.device, non_blocking=True),
            self.dones[sample_idxs].to(self.device, non_blocking=True)
        )
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha

        self.priorities[data_idxs] = priorities
        self.max_priority = np.max(self.priorities)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'

# Avoid Diamond Inheritance
class PrioritizedNStepReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, action_size, device, seed):
        ############################
        # YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################
        
    def __repr__(self) -> str:
        return f'Prioritized{self.n_step}StepReplayBuffer'

    def add(self, transition):
        ############################
        # YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################

    # def the other necessary class methods as your need
