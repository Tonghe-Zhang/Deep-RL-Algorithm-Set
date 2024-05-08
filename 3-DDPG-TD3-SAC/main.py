import hydra
import utils
import torch
import logging
import gymnasium as gym
from omegaconf import OmegaConf
from dotmap import DotMap
from hydra.utils import instantiate
from gymnasium.wrappers import RecordEpisodeStatistics
from core import train
from buffer import get_buffer
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    env = RecordEpisodeStatistics(gym.make(cfg.env_name, render_mode="rgb_array"))
    state_size = utils.get_space_shape(env.observation_space) # state size  == 8, 
    action_size = utils.get_space_shape(env.action_space)     # action size ==2, env.action_space==Box(-1.0, 1.0, (2,), float32)
    log_dict = utils.get_log_dict(cfg.agent._target_)
    for seed in cfg.seeds: #0,42,3407

        utils.set_seed_everywhere(env, seed)

        buffer = get_buffer(cfg.buffer, state_size=state_size, action_size=action_size, device=device, seed=seed)
        agent = instantiate(cfg.agent, state_size=state_size, action_size=action_size,
                            action_space=env.action_space, device=device) #env.action_space==Box(-1.0, 1.0, (2,), float32)
        
        logger.info(f"Training seed {seed} for {cfg.train.timesteps} timesteps with {agent} and {buffer}")

        # get_attr of omega_conf is slow, so we convert it to dotmap
        train_cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
        # return the mean and std of returns during training. eval_mean is a binary tuple of lists. 
        eval_mean = train(train_cfg, env, agent, buffer, seed, log_dict)
        logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")


if __name__ == "__main__":
    main()

'''
About the environment:

Action Space: Discrete(4)
    0: do nothing

    1: fire left orientation engine

    2: fire main engine

    3: fire right orientation engine

Observation Space: Box([-1.5 -1.5 -5. -5. -3.1415927 -5. -0. -0. ], [1.5 1.5 5. 5. 3.1415927 5. 1. 1. ], (8,), float32)
    The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, 
    its angle, its angular velocity, 
    and two booleans that represent whether each leg is in contact with the ground or not.
'''

