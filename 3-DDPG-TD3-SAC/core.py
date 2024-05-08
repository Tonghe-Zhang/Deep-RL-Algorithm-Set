import logging
import numpy as np
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from copy import deepcopy
from utils import merge_videos, visualize
from gymnasium.wrappers import RecordVideo
logger = logging.getLogger(__name__)
from tqdm import tqdm as tqdm

def eval(env, agent, episodes, seed):
    '''
    test the agent using three seeds, return the mean and standard deviation.

    episode: is specified in the config.yaml file, it is currently eval_episodes: 5
    this is the number of rollout trajectories used to calculte the mean and standard deviation
    of the accumulated reward of current policy.
    '''
    returns = []
    for episode in range(episodes):

        state, _ = env.reset(seed=episode + seed)

        done, truncated = False, False
        while not (done or truncated):
            state, _, done, truncated, info = env.step(agent.get_action(state))

        returns.append(info['episode']['r'].item())
    return np.mean(returns), np.std(returns)

def train(cfg, env, agent, buffer, seed, log_dict):
    eval_env = deepcopy(env)
    for key in log_dict.keys():
        log_dict[key].append([])
    
    done, truncated, best_reward = False, False, -np.inf
    state, _ = env.reset(seed=seed)
    # there are in total 200_000 times steps for training.   cfg.timesteps==200_000, see config.yaml file.
    for step in tqdm(range(1, cfg.timesteps + 1)):
        if done or truncated:
            state, _ = env.reset()
            done, truncated = False, False

            log_dict['train_returns'][-1].append(info['episode']['r'].item())
            log_dict['train_steps'][-1].append(step - 1)

        action = agent.get_action(state, sample=True)

        next_state, reward, done, truncated, info = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))
        state = next_state

        if step > cfg.batch_size + cfg.nstep:
            if isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch, weights=weights)
                buffer.update_priorities(tree_idxs, ret_dict['td_error'])

            elif isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch)
            else:
                raise RuntimeError("Unknown buffer")

            for key in ret_dict.keys():
                log_dict[key][-1].append(ret_dict[key])

        # we evaluate every 5000 steps, see config.yaml file.
        if step % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
            log_dict['eval_steps'][-1].append(step - 1)
            log_dict['eval_returns'][-1].append(eval_mean)
            logger.info(f"Seed: {seed}, Step: {step}, Eval mean: {eval_mean}, Eval std: {eval_std}")
            # we gonna save the best seed that yields the highest average reward to date.
            if eval_mean > best_reward:
                best_reward = eval_mean
                agent.save(f'best_model_seed_{seed}')

        # we also plot every 5000 steps, see config.yaml file.
        if step % cfg.plot_interval == 0:
            visualize(step, f'{agent} with {buffer}', log_dict)
            
    # save the final model.
    agent.save(f'final_model_seed_{seed}')

    # visualize the entire training process using log information accumulated in log_dict.
    visualize(step, f'{agent} with {buffer}', log_dict)

    env = RecordVideo(eval_env, f'final_videos_seed_{seed}', name_prefix='eval',
                      episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes,
                      disable_logger=True) # episode_trigger: this is a boolean function handle.

    # evalute the agent's performace.
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)

    # use best model so far for visualization
    agent.load(f'best_model_seed_{seed}')  
    env = RecordVideo(eval_env, f'best_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
    merge_videos(f'final_videos_seed_{seed}')
    merge_videos(f'best_videos_seed_{seed}')
    
    # end of simulation.
    env.close()

    return eval_mean    # including mean and std of all the episodes' returns.
