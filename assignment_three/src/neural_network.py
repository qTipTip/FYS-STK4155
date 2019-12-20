import torch.nn as nn
import torch.nn.functional as F


class GenericNN(nn.Module):

    def __init__(self, num_input_features=10, num_output_features=10, num_hidden_layers=1, num_hidden_features=20,
                 activation=F.relu):
        super().__init__()

        self.input_layer = nn.Linear(num_input_features, num_hidden_features).double()
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_hidden_features, num_hidden_features).double() for i in range(num_hidden_layers)])
        self.output_layer = nn.Linear(num_hidden_features, num_output_features).double()
        self.f = activation

    def forward(self, x):
        x = self.f(self.input_layer(x))
        for l in self.hidden_layers:
            x = self.f(l(x))
        x = self.output_layer(x)
        return x
