import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):

        super(DoubleQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, fc1_units)

        self.fc1_adv = nn.Linear(in_features=64, out_features=512)
        self.fc1_val = nn.Linear(in_features=64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=action_size)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))

        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.action_size)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)

        return x
