import numpy as np
import torch
import torch.nn.functional as F
from reinforce_algo import Algo
from reinforce_config import Config
from dataclasses import dataclass

@dataclass
class SampleData:
    state: int
    action: int
    reward: float

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(State).to(device)
        return tensor_state
            
class Agent:
    def __init__(self,env):
        torch.manual_seed(0)
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = 128
        self.action_dim = env.action_space.n
        self.learning_rate = 1e-3
        self.device = 'cuda'
        self.model = PolicyNet(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr = self.learning_rate)

    def take_action(self,state):
        s = torch.tensor(state).view(1, self.state_dim).to(self.device)
        # action_prob = self.model(s).detach().cpu().numpy().flatten()
        # action = np.random.choice(len(action_prob), p=action_prob)
        # return action
        action_dist = torch.distributions.Categorical(self.model(s))
        action = action_dist.sample()
        return action.item()

    def update(self,Episode):
        # 创建一个 Data 实例
        list_sample_data = [SampleData(state=obs, action=action, reward=r) for (obs,action,r) in Episode]
        algo = Algo(model = self.model, config = Config,  optimizer = self.optimizer, device = self.device)
        algo.learn(list_sample_data)
    
    def best_action(self,state):
        s = torch.tensor(state).view(1, self.state_dim).to(self.device)
        action = np.argmax(self.model(s).detach().cpu().numpy())
        return action
    

    
