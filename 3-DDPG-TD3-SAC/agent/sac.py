import torch
from torch import Tensor
from agent.td3 import TD3Agent
from models import SoftActor, Critic
from copy import deepcopy

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

import torch.distributions as td

class SACAgent(TD3Agent):
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, lr_alpha, gamma,
                 tau, nstep, target_update_interval, log_std_min, log_std_max, device):
        
        self.critic_net = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic_net).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=lr_critic)

        self.critic_net_2 = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target_2 = deepcopy(self.critic_net_2).to(device)
        self.critic_optimizer_2 = torch.optim.AdamW(self.critic_net_2.parameters(), lr=lr_critic)

        # in SAC the actor net is completely different.
        # forward pass to the SoftActor Network returns the mean and log_std of the action.  
        # a_t \sim \mu_{\theta}(\cdot|s_t)
        self.actor_net = SoftActor(state_size, 
                                   action_size, 
                                   hidden_dim, 
                                   deepcopy(action_space), 
                                   log_std_min, 
                                   log_std_max).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=lr_actor)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=lr_alpha)

        self.tau = tau
        self.device = device
        self.gamma = gamma ** nstep

        self.target_update_interval = target_update_interval
        
        # set the ``target_entropy'' 
        # this is actually the dimension of the continuous action space.
        self.target_entropy = \
        -torch.prod(torch.Tensor(action_space.shape).to(device))

        self.train_step = 0

    def __repr__(self):
        return 'SACAgent'

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch
        critic_loss, critic_loss_2, td_error = self.update_critic(state, action, reward, next_state, done, weights)
        actor_loss, alpha = self.update_actor(state)
        if not self.train_step % self.target_update_interval:
            self.soft_update(self.critic_target, self.critic_net)
            self.soft_update(self.critic_target_2, self.critic_net_2)
        self.train_step += 1
        return {'critic_loss': critic_loss, 'critic_loss_2': critic_loss_2, 'actor_loss': actor_loss, 'alpha': alpha, 'td_error': td_error}

    @jaxtyped(typechecker=beartype)
    def get_Qs(self, 
            state: Float[Tensor, "batch_size state_dim"], 
            action: Float[Tensor, "batch_size action_dim"], 
            reward: Float[Tensor, "batch_size"], 
            next_state: Float[Tensor, "batch_size state_dim"], 
            done: Int[Tensor, "batch_size"]
        ) -> tuple[Float[Tensor, "batch_size"], Float[Tensor, "batch_size"], Float[Tensor, "batch_size"]]:
        """
        Obtain the two Q value estimates and the target Q value 
        from the twin Q networks.
        """
        ############################
        Q=self.critic_net(state,action)
        Q2=self.critic_net_2(state,action)

        # since Q_target is computed from both Q and Q2, when
        # doing gradient descent for the two nets, repeated gradient computation error will occur. 
        # for this reason we will add no_grad() to all the computation of Q_target, treating it as 
        # a constant, while keeping Q and Q2 as two learnable networks with separate optimizers. 
        # then we use two lossses and two optimizers to adjust the twin nets, see function ``update_critic''. 
        with torch.no_grad():
            next_action, log_prob=self.actor_net.evaluate(state,sample=True)

            # Q_\theta'=r+\gamma*(1-done)*(min_{i=1,2}Q_{\theta'_i}(s',a')-alpha log \pi_\theta(a'|s'))
            Q_target=\
            reward+self.gamma*(
                torch.min(
                    self.critic_target(next_state,next_action),
                    self.critic_target_2(next_state,next_action))
                -(self.log_alpha.exp())*log_prob
            )*(torch.ones_like(done)-done)
        ############################
        return Q, Q2, Q_target
    
    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q, Q2, Q_target = self.get_Qs(state, action, reward, next_state, done)
        with torch.no_grad():
            td_error = torch.abs(Q - Q_target)
    
        if weights is None:
            critic_loss = torch.mean((Q - Q_target)**2)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2)
        else:
            critic_loss = torch.mean((Q - Q_target)**2 * weights)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2 * weights)

        # # original implemenation
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # revised version to 
        # self.critic_optimizer.zero_grad()
        # self.critic_optimizer_2.zero_grad()

        # critic_combined_loss=critic_loss+critic_loss_2
        # critic_combined_loss.backward()

        # self.critic_optimizer.step()
        # self.critic_optimizer_2.step()

        return critic_loss.item(), critic_loss_2.item(), td_error.mean().item()

    @jaxtyped(typechecker=beartype)
    def get_actor_loss(self, 
            state: Float[Tensor, "batch_size state_dim"]
        ) -> tuple[Float[Tensor, ""],
                   Float[Tensor, "batch_size"]]:
        """
        Calculate actor loss and log prob using policy network.

        outputs:
            actor_loss is used to optimize the policy network.
            action_log_prob is used to update the alpha factor.
        """
        ############################
        """
        the actor loss is different from td3 and ddpg. 
        we neeed to introduce an unbiased estimator of the entropy (which is the 
        negative log probability)
        and pick the minimum of the twin q nets to compute the policy gradient.
        
        maximize \min_{i=1,2} Q_{\phi_i}(s_t, \tilde{a}) + \alpha (-\log \pi_\theta(\tilde{a}|s_t))
        
        to use gradient descent for optimiazation we need to convet the signature, 
        which results in

        L_{\theta}= \alpha log_prob of the action  - \min_{i=1,2} Q_{\phi_i}(s_t, \tilde{a})
        """

        action, action_log_prob=self.actor_net.evaluate(state,sample=True)

        Q_min=torch.min(self.critic_net(state,action), self.critic_net_2(state,action))

        # the actor loss is defined as a float tesor, so we should take the mean over the batch of samples.
        # when optimizing the actor loss, we do not need to optimize alpha at the same time, 
        # because we optimized alpha by another loss. So we should detach() log_alpha during actor_loss's computation.

        actor_loss=(self.log_alpha.exp()* action_log_prob-Q_min).mean()
        ############################
        return actor_loss, action_log_prob
    
    def update_actor(self, state):

        actor_loss, action_log_prob = self.get_actor_loss(state)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha = self.update_alpha(action_log_prob)

        return actor_loss.item(), alpha.item()
    
    @jaxtyped(typechecker=beartype)
    def get_alpha_loss(self, 
            action_log_prob: Float[Tensor, "batch_size"]
        ) -> Float[Tensor, ""]:
        """     
        Calculate alpha loss.
        You should view `log_alpha` as alpha in the loss calculation here 
        to pass the autotest.

        here the input action probabilities are in batch, so our alpha loss will take the mean.
        the alpha loss is a single float tensor.
        L_\alpha = -\alpha (\log \pi_\theta(\cdot|s_t)+target_entropy)
        """
        ############################
        # since action_log_prob is associated with the actor net, whe should detach it to compute the alpha loss.
        # make sure that when computing the gradient w.r.t alpha, only the first term can be operated.
        # so you detach the rest.
        alpha_loss=-(self.log_alpha)*(action_log_prob+self.target_entropy).detach()
        alpha_loss=alpha_loss.mean()
        # print(alpha_loss)
        ##
        # since we use different optimizers for alpha and actor network, we do not need to wrirte .detach() 
        # after this expression. Notice that when calculating alpha_loss, the term actino_log_prob is 
        # outputu from the actor net, so if we do not use different optimizers the actor net will also be 
        # optimized, in this case we must add .detach() before .mean()
        ##
        ############################
        return alpha_loss
    
    def update_alpha(self, action_log_prob):
        alpha_loss = self.get_alpha_loss(action_log_prob.detach())

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return self.log_alpha.exp()

    @torch.no_grad()
    def get_action(self, state, sample=False):
        action, _ = self.actor_net.evaluate(torch.as_tensor(state).to(self.device), sample)
        return action.cpu().numpy()
