import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn, obs_unnormalization
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import RAMBOPolicy


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, adv-weight=3e-4
hopper-medium-v2: rollout-length=5, adv-weight=3e-4
walker2d-medium-v2: rollout-length=5, adv-weight=0
halfcheetah-medium-replay-v2: rollout-length=5, adv-weight=3e-4
hopper-medium-replay-v2: rollout-length=5, adv-weight=3e-4
walker2d-medium-replay-v2: rollout-length=5, adv-weight=0
halfcheetah-medium-expert-v2: rollout-length=5, adv-weight=0
hopper-medium-expert-v2: rollout-length=5, adv-weight=0
walker2d-medium-expert-v2: rollout-length=2, adv-weight=3e-4
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="rambo")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-adv-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=250)
    parser.add_argument("--dynamics-update-freq", type=int, default=1000)
    parser.add_argument("--adv-batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--adv-weight", type=float, default=3e-4)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--include-ent-in-adv", type=bool, default=False)
    parser.add_argument("--load-bc-path", type=str, default=None)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument("--bc-epoch", type=int, default=50)
    parser.add_argument("--bc-batch-size", type=int, default=256)

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    obs_mean, obs_std = real_buffer.normalize_obs()
    fake_buffer_size = args.step_per_epoch // args.rollout_freq * args.model_retain_epochs * args.rollout_batch_size * args.rollout_length
    fake_buffer = ReplayBuffer(
        buffer_size=fake_buffer_size, 
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    
    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    dynamics_adv_optim = torch.optim.Adam(
        dynamics_model.parameters(), 
        lr=args.dynamics_adv_lr
    )
    dynamics_scaler = StandardScaler()
    termination_fn = obs_unnormalization(get_termination_fn(task=args.task), obs_mean, obs_std)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        dynamics_scaler,
        termination_fn,
    )

    policy_scaler = StandardScaler(mu=obs_mean, std=obs_std)

    # create policy
    policy = RAMBOPolicy(
        dynamics, 
        actor, 
        critic1, 
        critic2, 
        actor_optim, 
        critic1_optim, 
        critic2_optim, 
        dynamics_adv_optim,
        tau=args.tau, 
        gamma=args.gamma, 
        alpha=alpha, 
        adv_weight=args.adv_weight, 
        adv_rollout_length=args.rollout_length, 
        adv_rollout_batch_size=args.adv_batch_size,
        include_ent_in_adv=args.include_ent_in_adv,
        scaler=policy_scaler,
        device=args.device
    ).to(args.device)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
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
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        dynamics_update_freq=args.dynamics_update_freq,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes
    )

    # train
    if args.load_bc_path:
        policy.load(args.load_bc_path)
        policy.to(args.device)
    else:
        policy.pretrain(real_buffer.sample_all(), args.bc_epoch, args.bc_batch_size, args.bc_lr, logger)
    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)
    else:
        dynamics.train(
            real_buffer.sample_all(),
            logger,
            holdout_ratio=0.1,
            logvar_loss_coef=0.001,
            max_epochs_since_update=10
        )

    policy_trainer.train()


if __name__ == "__main__":
    train()