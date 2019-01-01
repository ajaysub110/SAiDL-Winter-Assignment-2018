import torch
from torch import nn

class Agent(nn.Module):
    def __init__(self,din,dout):
        super(Agent,self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(din,64),
            nn.ReLU(True),
            nn.Linear(64,64),
            nn.ReLU(True),
            nn.Linear(64,dout)
        )
    
    def forward(self,obs):
        return self.NN(obs)