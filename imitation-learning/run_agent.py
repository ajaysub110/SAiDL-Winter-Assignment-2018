import os
import argparse
import pickle
import numpy as np
import gym
from train import load_model, save_model
from main import Opt
from agent import Agent
import torch
from torch.autograd import Variable

def main():
    opt = Opt()

    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of agent roll outs')
    args = parser.parse_args()

    print('loading and building agent policy')
    env = opt.env

    if opt.seed:
        env.seed(opt.seed)
        torch.manual_seed(opt.seed)

    model = Agent(env.observation_space.shape[0], env.action_space.shape[0]).double()
    load_model(opt,model,"./models/")
    print('loaded and built')

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = Variable(torch.from_numpy(obs))
            action = model.forward(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action.detach().numpy())
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()