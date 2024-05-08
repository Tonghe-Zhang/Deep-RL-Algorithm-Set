import torch
from torch import Tensor
from models import Critic
from copy import deepcopy
from agent.ddpg import DDPGAgent

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

class TD3Agent(DDPGAgent):
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep,
                 target_update_interval, noise_clip, policy_noise, policy_update_interval, eps_schedule, device):
        super().__init__(state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep, target_update_interval, eps_schedule, device)
        
        # for the twin value network 
        self.critic_net_2 = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target_2 = deepcopy(self.critic_net_2).to(device)
        self.critic_optimizer_2 = torch.optim.AdamW(self.critic_net_2.parameters(), lr=lr_critic)

        # for insertion of noise into the deterministic policy network
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        # for delayed policy update
        self.policy_update_interval = policy_update_interval

    def __repr__(self):
        return "TD3Agent"

    @jaxtyped(typechecker=beartype)
    def get_Qs(self, 
            state: Float[Tensor, "batch_size state_dim"], 
            action: Float[Tensor, "batch_size action_dim"], 
            reward: Float[Tensor, "batch_size"], 
            next_state: Float[Tensor, "batch_size state_dim"], 
            done: Int[Tensor, "batch_size"]
        ) -> tuple[Float[Tensor, "batch_size"],
                   Float[Tensor, "batch_size"],
                   Float[Tensor, "batch_size"]]:
        """
        Obtain the two Q value estimates and the target Q value from the twin Q networks.
        Hint: remember to use target policy smoothing.
        """
        ############################
        
        # Q-networks
        Q=self.critic_net(state,action)
        Q2=self.critic_net_2(state,action)

        # target policy smoothing. Notice that DDPG uses epsilon-scheduling noise, while 
        # in td3 we use noise with fixed std but clipped value
        # remember that since getQs function's output will be directly used to update the two critic networks, 
        # we should detach() or no_grad() all the varibles related to the actor nets during Q-function calculation.
        with torch.no_grad():
            
            deterministic_action=self.actor_target(next_state)

            noise=torch.clamp(torch.randn_like(deterministic_action)*self.policy_noise, -self.noise_clip, self.noise_clip)

            action_lower_bound=self.actor_net.action_space.low

            action_upper_bound=self.actor_net.action_space.high

            next_action=torch.clamp(deterministic_action+noise, action_lower_bound, action_upper_bound)
        
        # clipped double Q-learning
        Q_target=reward+self.gamma*(
            torch.min(
            self.critic_target(next_state,next_action), 
            self.critic_target_2(next_state,next_action)
            ))*(torch.ones_like(done)-done)
        ############################
        return Q, Q2, Q_target

    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q, Q2, Q_target = self.get_Qs(state, action, reward, next_state, done)
        with torch.no_grad():
            td_error = torch.abs(Q - Q_target)
    
        # compute the 2-norm td error for the twin nets and update their 
        # parameters respectively
        if weights is None:
            critic_loss = torch.mean((Q - Q_target)**2)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2)
        else:
            critic_loss = torch.mean((Q - Q_target)**2 * weights)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2 * weights)

        # # original implementation
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        # self.critic_optimizer_2.zero_grad()
        # critic_loss_2.backward()
        # self.critic_optimizer_2.step()

        # Revised implementation that avoide repeated compuation of gradients on the critic nets.
        self.critic_optimizer.zero_grad()
        self.critic_optimizer_2.zero_grad()

        critic_loss_combined=critic_loss+critic_loss_2
        critic_loss_combined.backward()

        self.critic_optimizer.step()
        self.critic_optimizer_2.step()

        return critic_loss.item(), critic_loss_2.item(), td_error.mean().item()
    
    def update(self, batch, weights=None):
        """
        reload the function inherited from ddpg.
        """
        state, action, reward, next_state, done = batch

        # update the twin Q nets every iteration.
        critic_loss, critic_loss_2, td_error = self.update_critic(state, action, reward, next_state, done, weights)

        log_dict = {'critic_loss': critic_loss, 'critic_loss_2': critic_loss_2, 'td_error': td_error}
        
        # perform delayed policy updates every self.policy_update_interval step, 
        # and add actor_loss to log_dict
        ############################
        if not self.train_step % self.policy_update_interval:
            # update actor network. Since this method is 
            # the method ``self.update_actor'' is borrowed from the ddpg agent, 
            # so we update actor network only based on the first Q network.
            actor_loss=self.update_actor(state) 
            log_dict.update({'actor_loss': actor_loss})
            
            # polyak averaging of the twin value nets.
            self.soft_update(self.critic_target, self.critic_net)
            self.soft_update(self.critic_target_2, self.critic_net_2)
            # polyak averaging of actor net. [## this line is newly added.##]
            self.soft_update(self.actor_target, self.actor_net)

        self.train_step += 1

        return log_dict
