import os
import torch
import platform
import numpy as np
import torch.optim as optim
from copy import deepcopy
from model import QNetwork, DuelingQNetwork

class DQNAgent:
    def __init__(self, state_size, action_size, cfg, device='cuda', compile=True):
        self.device = device
        self.use_double = cfg.use_double
        self.use_dueling = cfg.use_dueling
        self.target_update_interval = cfg.target_update_interval       
        '''
            q_model is the behavior policy net, which corresponds to $\theta$
            $target_net$ is $\theta^-$
        '''
        q_model = DuelingQNetwork if self.use_dueling else QNetwork
        self.q_net = q_model(state_size, action_size, cfg.hidden_size, cfg.activation).to(self.device)
        self.target_net = deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=cfg.lr)
        '''
        \tau is the blending ratio of the online network to the target network.
        '''
        self.tau = cfg.tau
        self.gamma = cfg.gamma ** cfg.nstep
        '''
        we raise \gamma to the power of .nstep because we want to incorporate the N-step return trick. 
        in N step return, the Q-function is updated via
        
        Y_t^n=\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_a Q(s_{t+n}, a;w-)      

        which replaces the normal update rule of
        Q(s,a)=R(s,a)+\gamma^1 Q(s,a;w-)

        so you can see that in Nstep setting, \gamma is replaced with \gamma^{n}, and n is specified by cfg.nstep in our application. 
        '''
        if platform.system() == "Linux" and compile:
            # torch.compile is not supported on Windows or MacOS
            self.compile()

    def compile(self):
        self.q_net = torch.compile(self.q_net)
        self.target_net = torch.compile(self.target_net)

    def soft_update(self, target, source):
        '''
        \theta^- = \tau*\theta + (1-\tau) \theta^-
        '''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def get_Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ############################
        '''
        This is the behavior policy network, which corresponds to parameter \theta.
        inputs:(batch input)
            state,    (N,4)
            action,   (N)
        outputs:
            Q:        (N,1)
        '''
        Qs=self.q_net(state)  # Qs.shape:  (N,2)
        act_id=action.unsqueeze(-1).expand(-1,2)   # rescale action to (N,2)
        Q=torch.gather(input=Qs,dim=1,index=act_id)[:,0]   # retrieve the corresponding Q(s,a) and only keep one copy. (N)
        ############################
        return Q
    
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        inputs: state: (N,4) numpy ndarray.
        output: actions, (N,)
        Get the optimal action according to the current Q value and state
        """
        ############################
        '''
        a=\argmax_{a} Q(s,a)
        print(f"shape of torch.tensor(state,dtype=torch.float)=={torch.tensor(state,dtype=torch.float).shape}")
        print(f"shape of self.q_net(torch.tensor(state,dtype=torch.float))=={self.q_net(torch.tensor(state,dtype=torch.float)).shape}")
        print(f"shape of self.q_net(torch.tensor(state,dtype=torch.float)).argmax(dim=1)=={self.q_net(torch.tensor(state,dtype=torch.float)).argmax(dim=1).shape}")
        '''
        action=self.q_net(torch.tensor(state,dtype=torch.float)).argmax(dim=-1)
        ############################
        return action

    @torch.no_grad()
    def get_Q_target(self, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Get the target Q value according to the Bellman equation
        shape: denote by N the batch size
        inputs:
            reward: torch.Size([N])
            done: torch.Size([N])  
            next_state: torch.Size([N,4]) , where 4 is the size of each state vector
        outputs:
            expected_output: torch.Size([N])
        """
        if self.use_double:
            ##########################
            '''
            using double q nets.
            Q(s,a;\theta^-) = r(s,a) + \gamma Q(s', \arg\max_{a'}Q(s', a'; \theta) ;\theta^-)
            output size: (N)
            '''
            max_id=self.q_net(next_state).argmax(dim=-1).unsqueeze(-1).expand(-1,2)   # max_id.shape: torch.Size([N,2])
            Q_target=reward+self.gamma*(torch.gather(input=self.target_net(next_state), dim=1, index=max_id)[:,0])*(torch.ones_like(done)-done)
            ##########################
        else:
            ##########################
            '''
            Updating target net on it's own. Could incur high overestimation bias.
            Notice that even if self.use_double is false, we still adopts target net, whose parameters lags behind q_net.
            When we update Q_target, we use target_net but not q_net to select actions.
            Q(s,a;\theta^-)= r(s,a) + \gamma max_a Q(s',a ;\theta^-)
            here .max(dim=1) means taking max_a  
            [0] picks the max values instead of the armax indices
            output size: (N)
            '''
            Q_target=reward+self.gamma*(self.target_net(next_state).max(dim=-1)[0])*(torch.ones_like(done)-done)
            # we remark that the last * is an Hadamard product.
            ##########################
        return Q_target

    def update(self, batch:tuple, step:int, weights=None):
        update_debug=False
        '''
        inputs:
            batch:   tuple of 5 tensors.  (state, action, reward, next_state, done)
                     each tensor is of shape torch.Size([N]), where N is the batch size e.g. 128.      
            weights: only not None when using Prioritized Replay Buffer. 
                     size: torch.Size([N])
        '''
        state, action, reward, next_state, done = batch

        Q_target = self.get_Q_target(reward, done, next_state)

        Q = self.get_Q(state, action)

        '''
        Q_target and Q are both of shape torch.Size([N])
        '''

        if weights is None:
            weights = torch.ones_like(Q).to(self.device)

        td_error = torch.abs(Q - Q_target).detach()

        if update_debug==True:    
            print(f"Q:{Q.shape}")
            print(f"Q_target:{Q_target.shape}")
            print(f"weights:{weights.shape}")
        loss = torch.mean((Q - Q_target)**2 * weights)
        
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # periodically update the target network.
        if not step % self.target_update_interval:
            self.soft_update(self.target_net, self.q_net)

        return loss.item(), td_error, Q.detach().mean().item()

    def save(self, name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join('models', name))

    def load(self, root_path='', name='best_model.pt'):
        self.q_net.load_state_dict(torch.load(os.path.join(root_path, 'models', name)))

    def __repr__(self) -> str:
        use_double = 'Double' if self.use_double else ''
        use_dueling = 'Dueling' if self.use_dueling else ''
        prefix = 'Normal' if not self.use_double and not self.use_dueling else ''
        return use_double + use_dueling + prefix + 'QNetwork'
