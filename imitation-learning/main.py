import gym
import torch
from load_policy import load_policy
from agent import Agent
from train import behavioral_cloning, dagger, Eval

class Opt():
    seed = 3
    envname = 'Humanoid-v2'
    env = gym.make(envname)
    method = 'DA' # BC: Behavioral Cloning   DA: DAgger
    device = torch.device('cpu')
    expert_path = './experts/'
    model_save_path = './models/'
    n_expert_rollouts = 30 # number of rollouts from expert
    n_dagger_rollouts = 10 # number of new rollouts from learned model for a DAgger iteration
    n_dagger_iter = 10 # number of DAgger iterations
    n_eval_rollouts = 10 # number of rollouts for evaluating a policy
    L2 = 0.00001
    lr = 0.0001
    epochs = 20
    batch_size = 64

    eval_steps = 500
    


def main():
    opt = Opt()
    print('*' * 20, opt.envname, opt.method, '*' * 20)
    env = opt.env
    if opt.seed:
        env.seed(opt.seed)
        torch.manual_seed(opt.seed)
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0]).to(opt.device)
    expert = load_policy(opt.expert_path + opt.envname + '.pkl')
    method = opt.method

    if method == 'BC':
        agent = behavioral_cloning(opt, agent, expert)
    elif method == 'DA':
        agent = dagger(opt, agent, expert)
    else:
        NotImplementedError(method)

    
    avrg_mean, avrg_std = Eval(opt, expert)
    print('[expert] avrg_mean:{:.2f}  avrg_std:{:.2f}'.format(avrg_mean, avrg_std))
        
    avrg_mean, avrg_std = Eval(opt, agent)
    print('[agent] avrg_mean:{:.2f}  avrg_std:{:.2f}'.format(avrg_mean, avrg_std))

if __name__ == '__main__':
    main()