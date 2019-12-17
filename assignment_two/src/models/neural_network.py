import torch
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from torch import nn as nn
from torch.nn import functional as F


class GenericNN(nn.Module):

    def __init__(self, num_input_features=10, num_output_features=10, num_hidden_layers=1, num_hidden_features=20, activation=F.relu,
                 alpha=1, regularization='none'):
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


class RegularizedNeuralNetClassifier(NeuralNetClassifier):

    def __init__(self, module, alpha=1, regularizer='none', criterion=torch.nn.NLLLoss,
                 train_split=CVSplit(5, stratified=True),
                 classes=None,
                 *args,
                 **kwargs):
        self.alpha = alpha
        self.regularizer = regularizer

        if 'regularizer' in kwargs.keys():
            kwargs.pop('regularizer')
        if 'alpha' in kwargs.keys():
            kwargs.pop('alpha')

        super().__init__(module, *args, criterion=criterion, train_split=train_split, classes=classes, **kwargs)

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        loss = super().get_loss(y_pred, y_true, *args, **kwargs)

        if self.regularizer == 'none':
            pass
        elif self.regularizer == 'l1':
            loss += self.alpha * sum([w.abs().sum() for w in self.module_.parameters()])
        elif self.regularizer == 'l2':
            loss += self.alpha * sum([(w.abs() ** 2).sum() for w in self.module_.parameters()])
        return loss