from torch import nn
from torch import optim


def function_modele(nclass=13,lr=0.01):

    inputsize = 256*256
    nunit = 10

    #mlp
    model=nn.Sequential(
        nn.Linear(inputsize,nunit),
        nn.ReLU(),
        nn.Linear(nunit,nclass))

    optimizer = optim.SGD(model.parameters(), lr=lr)

    return model,optimizer

