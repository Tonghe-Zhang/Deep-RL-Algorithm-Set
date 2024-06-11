import argparse
import random
import gym
import d4rl
import numpy as np
import torch
import os
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, EnsembleCritic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import EDACPolicy
from utils import gta_to_dataset, dataset_to_transitions
from trim_merge import trim_dataset,merge_dataset
"""
suggested hypers

halfcheetah-medium-v2: num-critics=10, eta=1.0
hopper-medium-v2: num-critics=50, eta=1.0
walker2d-medium-v2: num-critics=10, eta=1.0
halfcheetah-medium-replay-v2: num-critics=10, eta=1.0
hopper-medium-replay-v2: num-critics=50, eta=1.0
walker2d-medium-replay-v2: num-critics=10, eta=1.0
halfcheetah-medium-expert-v2: num-critics=10, eta=5.0
hopper-medium-expert-v2: num-critics=50, eta=1.0
walker2d-medium-expert-v2: num-critics=10, eta=5.0
"""

"""
suggested hypers

halfcheetah-medium-v2: num-critics=10, eta=1.0
halfcheetah-medium-replay-v2: num-critics=10, eta=1.0
halfcheetah-medium-expert-v2: num-critics=10, eta=5.0

hopper-medium-replay-v2: num-critics=50, eta=1.0
hopper-medium-v2: num-critics=50, eta=1.0
hopper-medium-expert-v2: num-critics=50, eta=1.0

walker2d-medium-v2: num-critics=10, eta=1.0
walker2d-medium-replay-v2: num-critics=10, eta=1.0
walker2d-medium-expert-v2: num-critics=10, eta=5.0
"""

"""
cd /root/autodl-fs/OfflineRL-Kit-main
conda activate mujoco_py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
python run_edac.py --num-critics 10 --eta 1.0 --task halfcheetah-medium-replay-v0
python run_edac.py --num-critics 10 --eta 1.0 --task halfcheetah-medium-replay-v0
nohup python run_edac.py --num-critics 10 --eta 1.0 --task halfcheetah-medium-replay-v0 &

python run_edac.py --num-critics 10 --eta 1.0 --task halfcheetah-medium-replay-v0 --data_type mixture --synthetic_ratio 0.6
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="edac")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--num-critics", type=int, default=50)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=False)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--normalize-reward", type=bool, default=False)
    parser.add_argument("--epoch", type=int, default=1200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_type", type=str, default="trim")
    parser.add_argument("--trim_ratio", type=float, default=0.0)
    parser.add_argument("--synthetic_ratio", type=float, default=0.0)

    return parser.parse_args()


def train(args):
    # create env and dataset
    print(f"set up envirnoment.")
    env = gym.make(args.task)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]
    
    # seed
    print(f"set seed.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    
    # load dataset
    print(f"load dataset")
    dataset_options=['original','trim','synthetic','mixture']
    dataset_type=args.data_type
    synthetic_set_path="./halfcheetah-medium-replay-v0-weighted.npz"
    N_merged=  100000
    """
    original:  100899
    synthetic: 500000
    """
    if dataset_type not in dataset_options:
        raise ValueError(f"dataset type {dataset_type} must be witthin {dataset_options} !")
    if dataset_type == 'original':
        dataset = qlearning_dataset(env)
    elif dataset_type == 'synthetic':
        dataset=gta_to_dataset(synthetic_set_path, first_time_load=True)
    elif dataset_type=='trim':
        trim_ratio=args.trim_ratio
        dataset=trim_dataset(env_name=args.task,trim_pct=trim_ratio)
    elif dataset_type=='mixture':
        synthetic_ratio=args.synthetic_ratio
        synthetic_set=gta_to_dataset(synthetic_set_path, first_time_load=True)
        dataset=merge_dataset(synthetic_dataset=synthetic_set, origianl_dataset_name=args.task, synthetic_ratio =synthetic_ratio)
    if args.normalize_reward:
        mu, std = dataset["rewards"].mean(), dataset["rewards"].std()
        dataset["rewards"] = (dataset["rewards"] - mu) / (std + 1e-3)
    
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critics = EnsembleCritic(
        np.prod(args.obs_shape), args.action_dim, \
        args.hidden_dims, num_ensemble=args.num_critics, \
        device=args.device
    )
    
    # init as in the EDAC paper
    for layer in critics.model[::2]:
        torch.nn.init.constant_(layer.bias, 0.1)
    torch.nn.init.uniform_(critics.model[-1].weight, -3e-3, 3e-3)
    torch.nn.init.uniform_(critics.model[-1].bias, -3e-3, 3e-3)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = EDACPolicy(
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        eta=args.eta
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset['observations']),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["num_critics", "eta"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    )
    print(f"start training.")
    policy_trainer.train()

if __name__ == "__main__":
    args=get_args()
    print(f"Received the following configurations from console:{args}")
    train(args)