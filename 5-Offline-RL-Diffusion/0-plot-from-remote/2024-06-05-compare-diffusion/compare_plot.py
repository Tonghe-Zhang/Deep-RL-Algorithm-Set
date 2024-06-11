import pandas as pd
import matplotlib.pyplot as plt

import gym 

import argparse
import random

import gym
import d4rl

import numpy as np
import torch

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, EnsembleCritic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import EDACPolicy

from utils import gta_to_dataset, dataset_to_transitions


env=gym.make("halfcheetah-medium-replay-v0")

# Placeholder for the environment's method to normalize the score
def get_normalized_score(return_value):
    
    return env.get_normalized_score(return_value)*100

# Function to read and process CSV files
def process_csv(file_name):
    df = pd.read_csv(file_name, header=None, names=['index', 'step', 'normalized_return'])
    df['normalized_return'] = df['normalized_return'].apply(get_normalized_score)
    return df

# Process each CSV file
df_iql = process_csv('env-iql.csv')
df_cql = process_csv('env-cql.csv')
df_bcq = process_csv('env-bcq.csv')

# Plotting
plt.figure(figsize=(10, 6))

# Plot each algorithm's data
plt.plot(df_iql['step'], df_iql['normalized_return'], label='IQL', color='red')
plt.plot(df_cql['step'], df_cql['normalized_return'], label='CQL', color='green')
plt.plot(df_bcq['step'], df_bcq['normalized_return'], label='BCQ', color='blue')

# Reading the CSV data into a dataframe
data = pd.read_csv('policy_training_progress.csv')

# Function to format axis in 1e6 scale
def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)
# Plotting eval/normalized_episode_reward and its standard deviation
plt.plot(data['timestep'], data['eval/normalized_episode_reward'], label='EDAC', color='black')

# Adding labels and title
plt.xlabel('Step')
plt.ylabel('Normalized Episodic Return')
plt.title('Comparison of Episodic Return of Different Offline RL Algorithms')
plt.legend()

plt.savefig("./figs/compare_3.png")

# Show the plot
plt.show()

