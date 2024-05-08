import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.transforms import TanhTransform
from torch.distributions import Normal, TransformedDistribution

from typing import Optional
from jaxtyping import Float, jaxtyped
from beartype import beartype

def mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, activation=nn.ELU):
    '''
    Input the input dimension, dimension of all the hidden layers, output dimension
    and the activation function at the output layer, and the activation function in between each of the hidden layers
    Output an nn.Sequential object which is the Multi-Layer-Perceptron specified by the input parameters.

                input layer    hidden layer 1      hidden layer k      last hidden layer         output layer       
                |              |                       |                      |                  |
    input dim=3 |              |        act()          |     act()            |          act()   |      Id()       ouptut dimension=4
                |              |                       |                      |                  |
                                                       |                                         |
                                                       |
    '''
    sizes = [input_size] + list(layer_sizes) + [output_size]
    layers = []
    # defined each hidder layer as fully connected+activation
    # leaving the output layer as fully connected+Identity map 
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """
    You input state-action batches to the Critic network:
        state: Nx8 tensor
        action: Nx2 tensor
    and the network directly concatenate the batches of state vectors and action vectors 
    along second dimension, forming an Nx10 tensor, and then directly feed this tensor 
    to the MLP. So we typically view the critic as a map from $S\times A \to \R$
    
    Output is a Nx1 tensor.
    """
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.fcs = mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state, action):
        return self.fcs(torch.cat([state, action], dim=1)).squeeze()


class Actor(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 action_space, 
                 hidden_dims = [400, 300], 
                 output_activation=nn.Tanh):
        super(Actor, self).__init__()

        '''
        Continuous action space
        '''
        #action_space=env.action_space==Box(-1.0, 1.0, (2,), float32)
        self.action_space = action_space

        # get the lower bounds of the 2 d action vector:
        # action_space.low==array([-1., -1.], dtype=float32)
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        # get the upper bounds of the 2 d action vector:
        # action_space.high==array([1., 1.], dtype=float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        
        '''
        Fully connected layers: sates --> hidden layers --> actions
        use tanh as output activation
        '''
        self.fcs = mlp(state_dim, hidden_dims, action_dim, output_activation=output_activation)
    
    def _normalize(self, action):
        '''
        before normalization: action \in [-1,1]
        after normalization:  action \in [action_space.low, action_space.high]
        '''
        return (action + 1) * (self.action_space.high - self.action_space.low) / 2 + self.action_space.low
    
    def to(self, device):
        '''
        Move the super class (containinig the fully connected layers) and the newly defined action spaces to given device.
        '''
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        return super().to(device)

    def forward(self, x):
        # use tanh as output activation
        # output: Nx2 tensor, the batch of deterministic actions for each input state vector.
        return self._normalize(self.fcs(x))


class SoftActor(Actor):
    def __init__(self, state_dim, action_dim, hidden_size, action_space, log_std_min, log_std_max):
        super().__init__(state_dim, 
                         action_dim * 2, 
                         action_space, 
                         hidden_dims=hidden_size, 
                         output_activation=nn.Identity)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    @jaxtyped(typechecker=beartype)
    def forward(self, 
            state: Float[Tensor, "*batch_size state_dim"]
        ) -> tuple[Float[Tensor, "*batch_size action_dim"],
                   Float[Tensor, "*batch_size action_dim"]]:
        """
        Obtain mean and log(std) from the fully-connected network.
        Crop the value of log_std to the specified range.
        
        We implement the mean and std of the action distribution as the outputs of the 
        neural net, which is learnable:

        \mathcal{N}(\mu_\theta, \sigma_\theta^2). 

        The actor net outputs Nx(2*dim_action_space), as is specified by the network design, 
        see __init__ method of 
        the SoftActor model. Notice that we defined the MLP's output dimension as action_dim * 2, 
        so the MLP becomes dim_state -> 2* dim_action, and the two output channels correspond to 
        the mean (of dimension ``dim_action'') and log(standard deviation) (also of dimension ``dim_action'')
        of the policy distribution's gaussian kernel. 
        Notice that the action is a 2D real-valued vector in LunarLanderContinuous environment, 
        so the mean and log_std deviation are both in \R^{2}.
        
        The reason that we take the output of the neural network as the logarithm of the std but not 
        the std, is because std is positive, but the neural net outputs to \R. 
        So we view the output as log(std), which picks value in \R (this shows some merit of the log funciton: \log(\cdot): \R_+ \to \R)

        After retrieving the log_std, we should take torch.exp() and then use the std.
        """
        ############################
        output=self.fcs(state)   # of dimension 2*action_space.

        action_dim=output.shape[-1]//2

        if output.dim()==1:   # in case batch size is None, when the replay buffer is not yet filled.
            mean=output[:action_dim]
            log_std=output[action_dim:]
        elif output.dim()==2:
            mean=output[:,:action_dim]
            log_std=output[:,action_dim:]

        log_std=torch.clamp(log_std,self.log_std_min,self.log_std_max) 

        # print(f"output.shape=={output.shape}, mean.shape=={mean.shape}, log_std.shape=={log_std.shape}")
        ############################
        return mean, log_std

    @jaxtyped(typechecker=beartype)
    def evaluate(self, 
            state: Float[Tensor, "*batch_size state_dim"],
            sample: bool = True
        ) -> tuple[Float[Tensor, "*batch_size action_dim"],
                   Optional[Float[Tensor, "*batch_size"]]]:
        
        mean, log_std = self.forward(state)
        # if we do not sample then it is equivalent to sampling at the mean.
        if not sample:
            return self._normalize(torch.tanh(mean)), None
        # sample means whether use reparameterization trick to sample from a stochastic policy.
        # sample action from N(mean, std) if sample is True
        # obtain log_prob for policy and Q function update
        # Hint: remember the reparameterization trick, and perform tanh normalization .
        """ 
        log_prob is the log probability of the new action: \log\pi_\theta(\tilde{a}^\prime|s^\prime). 
        It is used to optimize alpha and the actor network.
        """
        ############################
        """
        from torch.distributions.transforms import TanhTransform
        from torch.distributions import Normal, TransformedDistribution
        """
        normal_dist=Normal(mean,torch.exp(log_std))
        #this is a high dimensional gaussian distribution, with vector-valued mean and standard deviation
        # having dimension 2, which is the dimension of the action space

        """
        .rsample() is a method that enables gradient propagation from the sample drawn fromt the 
        parameterized distribution to these parameters represented by a neural network. 
        """
        raw_action=normal_dist.rsample()                   # a \sim N(mu,sigma^2)

        """
        We add tanh(\cdot): \R \to [-1,1] to the action (and thus the normal distribution) to 
        ensure that the output action is smoothly bounded within [-1,1], which 
        avoids extreme values of a and thus significantly improves the numerical stability. 
        
        Although adding tanh transform no longer insures that the action is sampled from a normalized distribution, 
        we can tolerate this bias because numerical stability is more important in practice. 
        However when returning the log probability of sampling this action, we need to add the transform correction:

        denote by f(\cdot):\R\to [-1,1] as tanh transform:   f(x)=e^{x}-e^{-x} / e^{x}+e^{-x}
        action : = Y = f(X) := tanh(raw_action), where raw_action \sim N(mu, \sigma^2). 
        But our output action follows tanh(N(mu,sigma^2)). 
        So the probability of observing the action   ``p_Y(y)'' can be calculated via

        p_X(\vec{x})=\norm{d\vec{y}/d\vec{x}}  p_Y{\vec{y}} 

        where \vec{x} and \vec{y} are vectors of the same dimenstion of the action space.

        Taking logarithm of either sides, we obtain

        log p(y) =  log p(x) - log (|dy/dx|)

        where |dy/dx| is the norm of the derivative of the tanh transform (y=f(x)=e^{x}-e^{-x} / e^{x}+e^{-x}), 
        which is |dy/dx|=1-tanh^2(x)=1-y^2  (this is easily obtained by elementary calculations )

        Then we have (mathemtically), 

        log p(y) =  log p(x) - log(1-y**2)

        In practical computation, we should be aware of the instability of log function, we should write:

        log_prob_of_actual_action = log_prob_of_gaussian_raw_action  - log (eps +  (1-y**2))

        where eps =1e-6 is a small positive constant to avoid log 0=NaN problem. 

        We can safely call .log_prob() method of the normal distribution class because it does log(eps+.) for you. 

        Since the action and raw actions are both batched vectors, we should be aware to sum up all the dimensions to obtain 
        log p(\vec{y}) = log \prod_{i=1}^{d} p(y_i)  = \sum_{i=1}^{d}  log p(y_i)

        Moreover, since the feasible action may not necessarily and exactly bounded in [-1,1], we 
        often add annother normalization to the action output to make it stay in the feasible range specified by the problem: [-a_{min}, a_{max}], 
        and this is exactly what self._normalize does.
        """
        # y=tanh(x)
        action=torch.tanh(raw_action)
        action=torch.as_tensor(self._normalize(action))    # in our setting ._normalize still places a in [-1,1]. 

        # return the log probability of the action chosen:  log \pi_\theta(\tilde{a}|s)
        # in particular, since action is a vector of dimension 2, 
        # log_prob is defined as the probability of taking this vector action, which should be a floating. 
        # the raw value obtained by accessing the log_prob distribution is naturally of dimension 2, 
        # since we assume different dimensions of actions are independent, we sum of all dimensions to obtain
        # \log p(\vec{v})= \sum_{i=1}^{d} \log p(\vec{v}_i) 
        # since the log_prob could be processed in batches, we only sum of the last dimension.

        # log p(y) =  log p(x) - log(1-y**2)
        eps=1e-6

        action_log_prob=normal_dist.log_prob(raw_action).sum(dim=-1)-torch.log(eps+(1-action**2)).sum(dim=-1)
        ############################
        return action, action_log_prob
    






    