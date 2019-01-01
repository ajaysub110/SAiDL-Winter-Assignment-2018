import os
import torch 
import numpy as np 
from torch import optim,nn
from torch.utils.data import TensorDataset,ConcatDataset,DataLoader

def agent_wrapper(opt, agent):
    def fn(obs):
        with torch.no_grad():
            obs = obs.astype(np.float32)
            assert len(obs.shape) == 2
            obs = torch.from_numpy(obs).to(opt.device)
            action = agent(obs)
        return action.cpu().numpy()
    return fn

def fit_dataset(opt,agent,dataset,num_epochs):
    optimizer = optim.Adam(agent.parameters(),lr=opt.lr,weight_decay=opt.L2)
    criterion = nn.MSELoss()
    dataloader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)

    step = 0
    best_reward = None 
    losses = []

    for i in range(num_epochs):
        for batch in dataloader:
            obs, gold_act = batch 
            pred_act = agent(obs)
            loss = criterion(pred_act,gold_act)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            losses.append(loss.item())

            if step % opt.eval_steps == 0:
                avrg_mean, avrg_std = Eval(opt, agent_wrapper(opt, agent))
                avrg_loss = np.mean(losses)
                losses = []
                print('[epoch {}  step {}] loss: {:.4f}  r_mean: {:.2f}  r_std: {:.2f}'.format(
                    i + 1, step, avrg_loss, avrg_mean, avrg_std))

                avrg_reward = avrg_mean - avrg_std
                if best_reward is None or best_reward < avrg_reward:
                    best_reward = avrg_reward
                    save_model(opt, agent, opt.model_save_path)
                
            step += 1     

    load_model(opt, agent, opt.model_save_path)

def behavioral_cloning(opt,agent,expert):
    expert_obs, expert_act, *_ = run_agent(opt,expert,opt.n_expert_rollouts)
    expert_obs = torch.from_numpy(expert_obs).to(opt.device)
    expert_act = torch.from_numpy(expert_act).to(opt.device)
    dataset = TensorDataset(expert_obs,expert_act)

    fit_dataset(opt,agent,dataset,opt.epochs)

    return agent_wrapper(opt,agent)

def dagger(opt,agent,expert):
    expert_obs, expert_act, *_ = run_agent(opt, expert, opt.n_expert_rollouts)
    expert_obs = torch.from_numpy(expert_obs).to(opt.device)
    expert_act = torch.from_numpy(expert_act).to(opt.device)
    dataset = TensorDataset(expert_obs, expert_act)

    for i in range(opt.n_dagger_iter):
        fit_dataset(opt,agent,dataset,opt.epochs)

        new_obs, *_ = run_agent(opt,agent_wrapper(opt,agent),opt.n_dagger_rollouts)
        expert_act = expert(new_obs)

        new_obs = torch.from_numpy(new_obs).to(opt.device)
        expert_act = torch.from_numpy(expert_act).to(opt.device)
        new_data = TensorDataset(new_obs, expert_act)

        dataset = ConcatDataset([dataset,new_data])

        avrg_mean, avrg_std = Eval(opt, agent_wrapper(opt, agent))
        print('[DAgger iter {}] r_mean: {:.2f}  r_std: {:.2f}'.format(i + 1, avrg_mean, avrg_std))

    return agent_wrapper(opt,agent)

def run_agent(opt,agent,num_rollouts):
    env = opt.env 
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    for i in range(num_rollouts):
        obs = env.reset()
        done = False 
        total_reward = 0
        steps = 0

        while not done:
            act = agent(obs[None,:])
            act = act.reshape(-1)
            observations.append(obs)
            actions.append(act)
            obs, reward, done, _ = env.step(act)
            total_reward += reward
            steps += 1
            if steps >= max_steps:
                break 
        returns.append(total_reward)
    
    avrg_mean, avrg_std = np.mean(returns), np.std(returns)
    observations = np.array(observations).astype(np.float32)
    actions = np.array(actions).astype(np.float32)

    return observations, actions, avrg_mean, avrg_std

def Eval(opt, agent):
    *_, avrg_mean, avrg_std = run_agent(opt, agent, opt.n_eval_rollouts)

    return avrg_mean, avrg_std


def save_model(opt, model, PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    PATH = PATH + opt.envname + '-' + 'parameters.tar'
    torch.save(model.state_dict(), PATH)
    print('model saved.')

def load_model(opt, model, PATH):
    PATH = PATH + opt.envname + '-' + 'parameters.tar'
    model.load_state_dict(torch.load(PATH))
    print('model loaded.')