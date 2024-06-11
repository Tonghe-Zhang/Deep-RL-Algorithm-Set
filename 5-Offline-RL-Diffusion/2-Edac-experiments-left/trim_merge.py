import d3rlpy
from utils import d3rlpy_episodes_to_dataset
import numpy as np
import gym
from offlinerlkit.utils.load_dataset import qlearning_dataset
from utils import dataset_to_episodes_syn,dataset_to_episodes_ori, d3rlpy_episodes_to_dataset, merge_dictionary
import random

def trim_dataset(env_name:str,trim_pct:float):
    """
    trim the original dataset. Remove the least ``trim_pct'' portion of the rewarding episodes. 
    return a dictionary dataset.
    """
    episodes, _ = d3rlpy.datasets.get_dataset(env_name)

    sorted_episodes=sorted(episodes, key = lambda episode: -episode.compute_return())

    top_percent_episodes=sorted_episodes[: int(len(sorted_episodes) * (1-trim_pct))]
    
    dataset = d3rlpy_episodes_to_dataset(top_percent_episodes)
    
    episodes_dict_list=dataset_to_episodes_ori(dataset)

    dataset=merge_dictionary(episodes_dict_list)
    
    print(f"len(dataset)={dataset['observations'].shape[0]}")
    
    return dataset

def merge_dataset(synthetic_dataset, origianl_dataset_name, synthetic_ratio = 0.6):
    
    original_episode_tuples, _ = d3rlpy.datasets.get_dataset(origianl_dataset_name)

    original_episodes=dataset_to_episodes_ori(d3rlpy_episodes_to_dataset(original_episode_tuples))

    synthetic_episodes=dataset_to_episodes_syn(synthetic_dataset)
    
    original_episode_length=original_episodes[0]['observations'].shape[0]
    
    synthetic_episode_length=synthetic_episodes[0]['observations'].shape[0]
    
    #print(f"number of original_episodes={len(original_episodes)}")
    #print(f"type(original_episodes)={type(original_episodes)}")
    #print(f"type(synthetic_episodes)={type(synthetic_episodes)}"
    print("%"*100)
    print("%"*100)
    
    print(f"Number of original episodes={len(original_episodes)}, each original episode contains {original_episode_length} transitions.")
    print(f"there are in total {len(original_episodes)*original_episode_length} original transitions.")
    
    print(f"Number of synthetic_episodes={len(synthetic_episodes)}, each synthetic episode contains {synthetic_episode_length} transitions.")
    print(f"There are in total {len(synthetic_episodes)*synthetic_episode_length} synthetic transitions.")
    
    merge_by='transitions'
    # Calculating number of samples to pick from each dataset based on the ratio
    if merge_by=='episodes':
        num_merged_episodes=len(synthetic_episodes)+len(original_episodes)
        num_synthetic_episodes = int(num_merged_episodes * synthetic_ratio)
        num_original_episodes = num_merged_episodes-num_synthetic_episodes
        N_syn=num_synthetic_episodes
        N_ori=num_original_episodes
        print(f"We will radomly shuffle {N_syn} synthetic episodes and {N_ori} original episodes, and then train on these transitions. Synthetic transition takes {synthetic_ratio*100} % of the merged dataset. ")
    elif merge_by=='transitions':
        num_merged_transitions=len(original_episodes)*original_episode_length
        # do not overflow. 
        num_synthetic_transitions=num_merged_transitions*synthetic_ratio
        num_originl_transitions=num_merged_transitions-num_synthetic_transitions
        N_syn=int(num_synthetic_transitions/synthetic_episode_length)
        N_ori=int(num_originl_transitions/original_episode_length)
        print("%"*100)
        print(f"We will radomly shuffle")
        print(f"{N_syn} episodes or {N_syn*synthetic_episode_length} transitions from the synthetic dataset")
        print(f"{N_ori} episodes or {N_ori*original_episode_length} transitions from the original dataset.")
        print(f"Synthetic transition ratio is set to be around {synthetic_ratio*100} %, actually it is {(N_syn*synthetic_episode_length/(N_syn*synthetic_episode_length+N_ori*original_episode_length))*100}%. ")
        print("%"*100)
    # Randomly selecting samples from each dataset
    indices1 = np.random.choice(np.arange(len(synthetic_episodes)), N_syn, replace=False)
    indices2 = np.random.choice(np.arange(len(original_episodes)), N_ori, replace=False)
    """
    list_of_episode_dict is a list of dictionaries, with each dictionary having five elements
    and each element is a numpy.ndarray of the observations, actions, etc in this episode.
    """
    list_of_episode_dict=[]
    for syn_episode_id in indices1:
        list_of_episode_dict.append(synthetic_episodes[syn_episode_id])
    for orig_episode_id in indices2:
        list_of_episode_dict.append(original_episodes[orig_episode_id])
    # we will randomly shuffle the episodes in the original and synthetic dataset to ensure
    # complete randomness. 
    random.shuffle(list_of_episode_dict)

    # we merge all the episodes into a single dictionary of dataset.
    dataset_merged=merge_dictionary(list_of_episode_dict)

    print(f"number of transitions in the merged dataset={len(dataset_merged['observations'])}")
    print("%"*100)
    print("%"*100)
    return dataset_merged

def merge_dataset_by_transition(synthetic_dataset, original_dataset, num_merged_dataset = 100000,synthetic_ratio = 0.6,print_data_merge=False):
    """
    Merges two datasets into a single dataset, with the synthetic dataset taking part of ``synthetic_ratio'' of the merged. 
    inputs:
        for original set, 
            N=1000899
            dim_obs=17
            dim_act=6
        for synthetic set, 
            N=5000021
        synthetic_dataset, original_dataset of the form:
            dataset is a dictionary of length 5, with the five elements being
            dataset['observations'] is a numpy.ndarray, .shape=[N, dim_obs]
            dataset['actions'] is a numpy.ndarray, .shape=[N, dim_act]
            dataset['next_observations'] is a numpy.ndarray, .shape=[N, dim_obs]
            dataset['rewards'] is a numpy.ndarray, .shape=(N,)  .shape[0] =N
            dataset['terminals'] is a numpy.ndarray, .shape=(N,)  .shape[0] =N
        num_merged_dataset: int, number of total samples trajectories in the merged set.
        synthetic_ratio: float, ratio of the synthetic dataset in the merged set.
    returns:
        dataset_merged, of the form above.
    """
    # Calculating number of samples to pick from each dataset based on the ratio
    num_samples1 = int(num_merged_dataset * synthetic_ratio)
    num_samples2 = num_merged_dataset - num_samples1

    N1=synthetic_dataset['observations'].shape[0]
    N2=original_dataset['observations'].shape[0]
    assert num_samples1 < N1 
    assert num_samples2 < N2 

    # Randomly selecting samples from each dataset
    indices1 = np.random.choice(np.arange(N1), num_samples1, replace=False)
    indices2 = np.random.choice(np.arange(N2), num_samples2, replace=False)

    # Merging the selected samples
    dataset_merged = {
        'observations': np.concatenate([synthetic_dataset['observations'][indices1], original_dataset['observations'][indices2]], axis=0),
        'actions': np.concatenate([synthetic_dataset['actions'][indices1], original_dataset['actions'][indices2]], axis=0),
        'next_observations': np.concatenate([synthetic_dataset['next_observations'][indices1], original_dataset['next_observations'][indices2]], axis=0),
        'rewards': np.concatenate([synthetic_dataset['rewards'][indices1], original_dataset['rewards'][indices2]], axis=0),
        'terminals': np.concatenate([synthetic_dataset['terminals'][indices1], original_dataset['terminals'][indices2]], axis=0)
    }

    if print_data_merge:
        pprint(synthetic_dataset)
        pprint(original_dataset)
        pprint(dataset_merged)

    return dataset_merged
