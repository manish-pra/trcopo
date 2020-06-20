import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(policy, self).__init__()
        # self.actor = nn.Sequential(nn.Linear(state_dim, action_dim),  # 2(self and other's position) * 2(action)
        #                            nn.Softmax(dim =-1))  # 20*2
        self.actor = nn.Sequential(nn.Linear(state_dim, 64),  # 84*50
                                   nn.Tanh(),
                                   nn.Linear(64, 32),  # 50*20
                                   nn.Tanh(),
                                   nn.Linear(32, action_dim),
                                   nn.Softmax(dim=-1))  #There is no variance parameter because it is a categorical distribution
    def forward(self, state):
        mu = self.actor(state)
        return mu

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic, self).__init__()

        self.critic = nn.Sequential(nn.Linear(state_dim + action_dim, 64),  # 84*50
                                    nn.Tanh(),
                                    nn.Linear(64, 32),  # 50*20
                                    nn.Tanh(),
                                    nn.Linear(32, 1))  # 20*2

        #self.apply(init_weights)

    def forward(self, state):
        value = self.critic(state)
        return value

class policy_orca(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(policy_orca, self).__init__()
        # self.actor = nn.Sequential(nn.Linear(state_dim, action_dim),  # 2(self and other's position) * 2(action)
        #                            nn.Softmax(dim =-1))  # 20*2
        self.actor = nn.Sequential(nn.Linear(state_dim, 128),  # 84*50
                                   nn.Tanh(),
                                   nn.Linear(128, 128),  # 50*20
                                   nn.Tanh(),
                                   nn.Linear(128, action_dim),
                                   nn.Softmax(dim=-1))  #There is no variance parameter because it is a categorical distribution
    def forward(self, state):
        mu = self.actor(state)
        return mu

class critic_orca(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic_orca, self).__init__()

        self.critic = nn.Sequential(nn.Linear(state_dim + action_dim, 128),  # 84*50
                                    nn.Tanh(),
                                    nn.Linear(128, 128),  # 50*20
                                    nn.Tanh(),
                                    nn.Linear(128, 1))  # 20*2

        #self.apply(init_weights)

    def forward(self, state):
        value = self.critic(state)
        return value