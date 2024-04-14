import torch
import torch.nn as nn
from hydra.utils import instantiate



'''
Description of the cart-pole v1 environment

Environment: see https://gymnasium.farama.org/environments/classic_control/cart_pole/

state:  np.ndarry  4 dim array   (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity)

action: The action is a ndarray with shape (1,) which can take values {0, 1}

The episode ends if any one of the following occurs:

Termination: Pole Angle is greater than ±12°

Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)

Truncation: Episode length is greater than 500 (200 for v0)

'''
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation):
        super(QNetwork, self).__init__()
        '''
        Remember that each state in the Cartpole-v1 is a 4dim array, so the input dim of the Q-network has dimension state_size=4
        see test.main.py
        '''
        self.q_head = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            instantiate(activation),   # this is the activation function.
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, action_size)
        )
    def forward(self, state):
        '''
        state should be of shape "state_size"
        in Q network, we input some state vector \vec{s}, and the output is an A-size array, which corresponds to  vector Q(s,\cdot): the table of Q values at the given(input) state and any actions.
        '''
        Qs = self.q_head(state)
        return Qs

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            instantiate(activation),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the Q value of the current state Q(state,\cdot) using dueling network
        """
        ############################
        '''
        shape:
            input:
                state: torch.Size([N,4])
            output:
                Qs:torch.Size([N,2])
        Formula:
            Q(s,a; \alpha,\beta, \theta)= V(s; \beta)  +    A(s,a;\alpha)-\frac{1}{\mathcal{A}} \sum_{a'} A(s,a';\alpha)
        '''
        # Note that we must feed the same feature map to the value and advantage head.
        # adopting different feature maps will distablize the training. Besides ,
        # a more complex model structure brings an increased risk of overfitting to the training data.
        # using a shared feature layer ensures that both streams work from a common understanding of the environment, which can simplify the learning task and improve the efficiency of the network.
        common_feature=self.feature_layer(state)
        

        # compute V(s; \alpha)
        value_function=self.value_head(common_feature)   # [N,1]    

        # compute A(s,\cdot; \alpha)
        advantage_function=self.advantage_head(common_feature) #[N,2]
        # very important step: though by mathematical definition the advantage is zero mean under policy \pi: \mathbb{E}_{a\sim \pi(\cdot|s)} A(s,a) = 0
        # it may not necessarily be so by using black-box neural nets. so we have to manually normalize it after inference.
        # rigorously speaking, we should use the current policy to normalize the advantage function as a weighted average. 
        # however in most PyTorch implementations this methos is a bit too complicated.
        # directly subtracting the mean also helps to stabalize training.
        advantage_function=advantage_function-torch.mean(advantage_function, dim=-1, keepdim=True) #[N,2]

        # compute Q(s,\cdot; \alpha)
        Qs=value_function+advantage_function
        ############################
        return Qs




