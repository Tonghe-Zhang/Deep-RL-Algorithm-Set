import torch
import random
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
    def __init__(self, capacity:int, state_size, seed, device):
        """
        The reply buffer kind of resembles to a recurrent queue in terms of data structure.
        It has a pointer(self.idx), capacity(self.capacity), and an effective-capacity recorder (self.size)
        Since we draw random trajectories from the reply buffer, we also have a built-in random number generator (self.rng)
        The buffer may be huge, so we may move it to gpu and we need to specify self.device.
        The buffer contains multiple arrays of (s,a,s',r,terminate)
        'seed':
            is the predefined random seed. We pass random seed as a parameter for reproducibility.
            in this application,  it is chosen as 3407 in the config.yaml file under the variable name of 'seeds'.
        'capacity':
            integer. the numeber of transitions in the replay buffer.
        'state_size':
            in our application 'cart-pole', 
            the state is an $\R^{4}$ vector containing the position, velocity, angle and angular velocity.
            so the states in the replay buffer is a little bits special and it contains more than one dimension. 
            however the actions and rewards are both 1-dimensional.
        """
        self.device = device

        self.rng = np.random.default_rng(seed)
        # 'rng' is the abbreviation for random number generator. 

        self.idx = 0
        # idx is a pointer recording the current number of valid samples.
        # idx is updated whenever a new data sample(trajectory) is put into the buffer.
        # when idx overflows it is rounded by capacity.

        self.size = 0
        # the effective size of the data loader, or the actual number of samples that can be used. 
        # equals min(self.capacity, self.size + 1)
        
        self.capacity = capacity
       
        # the storage of the buffer.
        self.states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous()#.pin_memory()

        self.actions = torch.zeros(capacity, dtype=torch.long).contiguous()#.pin_memory()

        self.rewards = torch.zeros(capacity, dtype=torch.float).contiguous()#.pin_memory()

        self.next_states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous()#.pin_memory()

        self.dones = torch.zeros(capacity, dtype=torch.int).contiguous()#.pin_memory()
        # terminate, boolean.

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        '''
        add a new transition into the replay buffer
        '''
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.states[self.idx] = torch.as_tensor(state)
        self.actions[self.idx] = torch.as_tensor(action)
        self.rewards[self.idx] = torch.as_tensor(reward)
        self.next_states[self.idx] = torch.as_tensor(next_state)
        self.dones[self.idx] = torch.as_tensor(done)

        # update counters
        '''
        you can see that we periodically iter over the buffer to add new samples. 
        if the buffer is full, then a new buffer is added to the first slot, replacing the oldest sample, so that 
        the buffer is refreshed in a periodical fashion.
        '''
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size:int)->tuple:
        '''
        generate a batch of samples. 
        inputs:
            batch_size, number of (s,a,r,s',d) pairs to be generated in a series of calls to sample() method, 
            which determines the range of the sampled id in each singel call of this method.
        output:
            a single five element tensor tuple (s,a,r,s',d)
        '''
        # sample a trajectory from replay buffer.
        # using np.random.default_rng().choice is faster https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/

        # randomly choose one id from the effective samples. 
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        # draw that transition and return it.
        batch = (
            self.states[sample_idxs].to(self.device, non_blocking=True),
            self.actions[sample_idxs].to(self.device, non_blocking=True),
            self.rewards[sample_idxs].to(self.device, non_blocking=True),
            self.next_states[sample_idxs].to(self.device, non_blocking=True),
            self.dones[sample_idxs].to(self.device, non_blocking=True)
        )
        return batch

class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, seed, device):
        super().__init__(capacity, state_size, seed, device=device)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
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
        return state, action, reward_look_ahead, done

    def add(self, transition):
        # we store the most recent transitions into the NStepBuffer until it is full. 
        state_record, action_record, reward_record, next_state, done_record = transition

        self.n_step_buffer.append((state_record, action_record, reward_record, done_record))

        if len(self.n_step_buffer) < self.n_step:
            return
        # when NStepBuffer is full, we retrieve the first (s,a), n-step accumulated reward, and the last s' 
        # and feed it to the agent to calculate the Bellman update.

        # state=s_t, action=a_t  reward_look_ahead=\sum_{k=0}^{n_step-1}\gamma^{k}R_{t+k+1}   done=if there is a done before s_{t+n} or s_{t+n} is done.
        state, action, reward_look_ahead, done = self.n_step_handler()
        # next_state is s_{t+n}
        super().add((state, action, reward_look_ahead, next_state, done))
        # Notice that we have inherited .add() method from super class 'ReplayBuffer', so .add() belongs to ReplayBuffer 


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, seed, device):
        '''
        capacity is the length of the buffer, which is 50,000 by default.
        '''
        # self.tree = SumTreeArray(capacity, dtype='float32')

        # self.priorities = np.zeros(capacity, dtype=np.float32)

        self.priorities = np.ones(capacity, dtype=np.float32)

        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, seed, device=device)

    def add(self, transition):
        self.priorities[self.idx] = self.max_priority

        super().add(transition)
        # whenever we add a transition, the attribute of the superclass self.size will add 1, and truncated at self.capacity. if 
        # the buffer is full.

    def sample(self, batch_size:int):
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
        batch = (
            self.states[sample_idxs].to(self.device, non_blocking=True),
            self.actions[sample_idxs].to(self.device, non_blocking=True),
            self.rewards[sample_idxs].to(self.device, non_blocking=True),
            self.next_states[sample_idxs].to(self.device, non_blocking=True),
            self.dones[sample_idxs].to(self.device, non_blocking=True)
        )
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities_update = (priorities + self.eps) ** self.alpha

        self.priorities[data_idxs] = priorities_update

        self.max_priority = max(self.priorities)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'

# Avoid Diamond Inheritance
class PrioritizedNStepReplayBuffer():
    # Implement the PrioritizedNStepReplayBuffer class if you want to, this is OPTIONAL
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, seed, device):
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