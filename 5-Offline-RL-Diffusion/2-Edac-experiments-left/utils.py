import numpy as np
from d3rlpy.dataset import Transition, MDPDataset
import os

def merge_dictionary(list_of_dict):
    merged_data = {}

    for d in list_of_dict:
        for k, v in d.items():
            if k not in merged_data.keys():
                merged_data[k] = [v]
            else:
                merged_data[k].append(v)

    for k, v in merged_data.items():
        merged_data[k] = np.concatenate(merged_data[k])

    return merged_data

def gta_to_dataset(gta_path, first_time_load=False):    
    """
    input a gta file path, extract the synthetic dataset.

    dataset is a dictionary of length 5, with the five elements being
        dataset['observations'] is a numpy.ndarray, .shape=[N, dim_obs]
        dataset['actions'] is a numpy.ndarray, .shape=[N, dim_act]
        dataset['next_observations'] is a numpy.ndarray, .shape=[N, dim_obs]
        dataset['rewards'] is a numpy.ndarray, .shape=(N,)  .shape[0] =N
        dataset['terminals'] is a numpy.ndarray, .shape=(N,)  .shape[0] =N
    """
    gta_data = np.load(gta_path, allow_pickle=True)
    config_dict = gta_data['config'].item()
    dataset = gta_data['data'].squeeze()
    data = merge_dictionary([*dataset])

    metadata = {}
    metadata['diffusion_horizon'] = config_dict['construct_diffusion_model']['denoising_network']['horizon']
    metadata['diffusion_backbone'] = config_dict['construct_diffusion_model']['denoising_network']['_target_'].split('.')[-1]
    metadata['conditioned'] = True if config_dict['construct_diffusion_model']['denoising_network']['cond_dim'] != 0 else False
    metadata['guidance_target_multiple'] = config_dict['SimpleDiffusionGenerator']['amplify_returnscale']
    metadata['noise_level'] = config_dict['SimpleDiffusionGenerator']['noise_level']
    
    if first_time_load:
        np.savez(
            f"./{gta_path.split('.')[-2]}-dataset.npz",
            observations=data['observations'],
            actions=data['actions'],
            rewards=data['rewards'],
            next_observations=data['next_observations'],
            terminals=data['terminals'],
        )
    return data


def dataset_to_episodes_ori(dataset):
    """
    input a dataset dictionary, 
    output a list of episodes. 
    Each episode is a dictionary of numpy.ndaarrays, which share the same format of a dataset.
    """
    episodes = []
    current_episode = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'terminals': []
    }
    #print(f"number of transitions in original dataset={len(dataset.terminals)}")
    #print(f"len(dataset.terminals)-1={len(dataset.terminals)-1}")
    #print(f"max(dataset.terminals)={max(dataset.terminals)}")
    #print(f"max(dataset.episode_terminals)={max(dataset.episode_terminals)}")
    for i in range(len(dataset.terminals)-1):
        current_episode['observations'].append(dataset.observations[i])
        current_episode['actions'].append(dataset.actions[i])
        current_episode['rewards'].append(dataset.rewards[i])
        current_episode['next_observations'].append(dataset.observations[i+1])
        current_episode['terminals'].append(dataset.episode_terminals[i])

        if bool(dataset.episode_terminals[i]):
            #print(f"i={i}, append an episode!")
            episodes.append({'observations': np.array(current_episode['observations']),
                            'actions':np.array(current_episode['actions']),
                            'rewards':np.array(current_episode['rewards']),
                            'next_observations':np.array(current_episode['next_observations']),
                            'terminals':np.array(current_episode['terminals'])})
            current_episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': []
            }
    if len(episodes) == 0:
        episodes=current_episode
        #print(f"none.")
    else:
        pass
        #print(f"have.")
    #print(f"Length of original episodes ={len(episodes)}")
    
    return episodes


def dataset_to_episodes_syn(dataset):
    """
    input a dataset dictionary, 
    output  a list of episodes. 
    Each episode is a dictionary of numpy.ndaarrays, which share the same format of a dataset.
    """
    episodes = []
    current_episode = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'terminals': []
    }
    
    #print(f"max(dataset['terminals'])={max(dataset['terminals'])}")
    #print(f"number of transitions in synthetic dataset={len(dataset['terminals'])}")
    
    for i in range(len(dataset['terminals'])):
        current_episode['observations'].append(dataset['observations'][i])
        current_episode['actions'].append(dataset['actions'][i])
        current_episode['rewards'].append(dataset['rewards'][i])
        current_episode['next_observations'].append(dataset['next_observations'][i])
        current_episode['terminals'].append(dataset['terminals'][i])

        if i>1 and i%30==29:
            #print(f"i={i}, append an episode!")
            episodes.append({'observations': np.array(current_episode['observations']),
                            'actions':np.array(current_episode['actions']),
                            'rewards':np.array(current_episode['rewards']),
                            'next_observations':np.array(current_episode['next_observations']),
                            'terminals':np.array(current_episode['terminals'])})
            current_episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': []
            }
    #print(f"number of synthetic episodes={len(episodes)}")
    return episodes


def d3rlpy_episodes_to_dataset(episodes):
    observations =  np.concatenate([episode.observations for episode in episodes], axis=0)
    actions = np.concatenate([episode.actions for episode in episodes], axis=0)
    rewards = np.concatenate([episode.rewards for episode in episodes], axis=0)
    terminals = np.zeros_like(rewards)
    episode_terminals = np.zeros_like(rewards)
    end_idxs = np.array([episode.observations.shape[0] for episode in episodes]).cumsum() - 1
    np.put_along_axis(terminals, end_idxs, np.array([episode.terminal for episode in episodes]), 0)
    np.put_along_axis(episode_terminals, end_idxs, 1, 0)

    mdp_dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
        episode_terminals=np.array(episode_terminals, dtype=np.float32),
    )
    return mdp_dataset



def dataset_to_transitions(dataset,):
    rets = []
    num_data = dataset['observations'].shape[0]    
    observation_shape = dataset['observations'][0].shape
    action_size = dataset['actions'][0].shape[0]
    
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']
    terminals = dataset['terminals']
    
    for i in range(num_data):
        transition = Transition(
            observation_shape=observation_shape,
            action_size=action_size,
            observation=observations[i],
            action=actions[i],
            reward=rewards[i],
            next_observation=next_observations[i],
            terminal=terminals[i],
        )
        # set pointer to the next transition
        rets.append(transition)
    return rets

