
class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, seed, device):
        super().__init__(capacity, state_size, seed, device=device)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step) 
        # deque of tuples (state, action, reward, done). Notice that we do not store next states like the main buffer. 
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        """Get n-step state, action, reward and done for the transition,
        discard those rewards after done=True"""

        '''
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

        # calculate accumulated future reward=\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} 
        state, action, _, done=self.n_step_buffer[0]
        reward_look_ahead=0
        for k, (_,_,r,d) in enumerate(self.n_step_buffer):         
            if d is True:
               done=d 
               break
            reward_look_ahead+=self.gamma**k*r

        return state, action, reward_look_ahead, done
    
    def add(self, transition):
        '''
        see line 45 in core.py 'buffer.add((state, action, reward, next_state, int(done)))'
        types:
            state: 4dim tensor
            action:1dim tensor
            reward:1dim tensor
            next_state:1dim tensor
            done:int
        '''
        
        # we store the most recent transitions into the NStepBuffer until it is full. 
        state_store, action_store, reward_store, next_state, done_store = transition
        
        self.n_step_buffer.append((state_store, action_store, reward_store, done_store))
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # when NStepBuffer is full, we retrieve the first (s,a), n-step accumulated reward, and the last s' 
        # and feed it to the agent to calculate the Bellman update.

        # state=s_t, action=a_t  reward_look_ahead=\sum_{k=0}^{n_step-1}\gamma^{k}R_{t+k+1}   done=if there is a done before s_{t+n} or s_{t+n} is done.
        state, action, reward_look_ahead, done = self.n_step_handler()
        # next_state is s_{t+n}
        super().add((state, action, reward_look_ahead, next_state, done))
        # Notice that we have inherited .add() method from super class 'ReplayBuffer', so .add() belongs to ReplayBuffer 
