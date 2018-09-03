#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from jaco_arm import JacoEnv, JacoCurEnv
import mujoco_py
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.dataset.mujoco_intertactive_dset import Mujoco_Dset as interactive_Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier

def train(env_id, num_timesteps, seed):
    from baselines.ppo_pnp import mlp_policy, pposgd_simple, interactive_ppo, ppo_gail_cur, ppo_interactive_ppo_cur
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3)
    env = JacoEnv(64, 64, 1, 1.0)#make_mujoco_env(env_id, seed)
    curEnv = JacoCurEnv(64, 64, 1, 1.0)#make_mujoco_env(env_id, seed)
    dataset = interactive_Mujoco_Dset(expert_path='data/pnp_demo.npz', traj_limitation=-1)#interactive_Mujoco_Dset(expert_path='data/lift_demo.npz', traj_limitation=-1)
    reward_giver = TransitionClassifier(env, 100, entcoeff=1e-3)
    ppo_interactive_ppo_cur.learn(env, curEnv, policy_fn, reward_giver, dataset,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()
    curEnv.close()

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=1000000000, seed=args.seed)

if __name__ == '__main__':
    main()
