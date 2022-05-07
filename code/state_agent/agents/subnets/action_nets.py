# Author: Jose Rojas (jlrojas@utexas.edu)
# Creation Date: 4/19/2022

from typing import Tuple
import torch
import numpy as np

def new_action_net(n_outputs=1, type="linear_tanh"):
    if type == "linear_sigmoid":
        return LinearWithSigmoid(n_outputs)
    elif type == "linear_tanh":
        return LinearWithTanh(n_outputs)
    else:
        raise Exception("Unknown action net")

class BaseNetwork(torch.nn.Module):

    def __init__(self, net, range: Tuple) -> None:
        super().__init__()       
        self.range = range
        self.net = net

    def get_input(self, x) -> torch.Tensor:
        if self.range is not None:
            if x.dim() == 1:                
                x = x[self.range[0]:self.range[1]]
            else:
                x = x[:,self.range[0]:self.range[1]]
        return x
           
    def forward(self, x):
        return self.net(self.get_input(x))

    
class SingleLinearNetwork(BaseNetwork):
    
    def __init__(self, n_inputs, n_outputs, bias) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.activation = None

        layers = [
            #torch.nn.BatchNorm1d(3*5*3),
            torch.nn.Linear(n_inputs, n_outputs, bias=bias),
            torch.nn.ReLU()            
        ]
        
        self.net = torch.nn.Sequential(
            *layers
        )
class LinearNetwork(torch.nn.Module):

    def __init__(self, activation, n_inputs, n_outputs, n_hidden, bias, scale, range) -> None:
        super().__init__()

        self.n_outputs = n_outputs
        self.scale = torch.Tensor(scale if scale else [1.0])
        self.activation = activation.__name__ if activation else None

        layers = [
            #torch.nn.BatchNorm1d(3*5*3),
            torch.nn.Linear(n_inputs, n_hidden, bias=bias),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_outputs, bias=bias)
        ]
        if activation:
            layers.append(activation())

        net = torch.nn.Sequential(
            *layers
        )

        self.base = BaseNetwork(net, range)


    def forward(self, x):
        return self.base.forward(x) * self.scale

class LinearWithSigmoid(LinearNetwork):

    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=20, bias=False, hard=False, **kwargs) -> None:
        super().__init__(torch.nn.Sigmoid if not hard else torch.nn.Hardsigmoid, n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, bias=bias, **kwargs)
    
class LinearWithTanh(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden, bias, scale, range) -> None:
        super().__init__()
        self.activation = "Tanh"
        self.linear = LinearNetwork(activation=torch.nn.Tanh, n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, bias=bias, scale=scale, range=range)

    def forward(self, x):
        if self.training:
            output = self.linear(x)
            # the training output needs to be a probability
            output = (output + 1) / 2
            return output * self.linear.scale
        else:
            output = self.linear(x)
            return output

class LinearWithSoftmax(LinearNetwork):

    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=20, bias=False, **kwargs) -> None:        
        super().__init__(torch.nn.Softmax, n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, bias=bias, **kwargs)
    
class LinearForNormalAndStd(LinearNetwork):

    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=20, bias=False, **kwargs) -> None:        
        # need double the number of outputs for mean and std        
        super().__init__(torch.nn.Tanh, n_inputs=n_inputs, n_outputs=n_outputs*2, n_hidden=n_hidden, bias=bias, **kwargs)

    def forward(self, x):
        output = super().forward(x)

        # force the std to be positive
        if x.dim() == 1:            
            return torch.concat([output[0:self.n_outputs//2], torch.abs(output[self.n_outputs//2:self.n_outputs])])
        else:
            return torch.concat([output[:, 0:self.n_outputs//2], torch.abs(output[:, self.n_outputs//2:self.n_outputs])], dim=1)

class BooleanClassifier(LinearWithTanh):

    def __init__(self, **kwargs) -> None:        
        super().__init__(n_outputs=1, **kwargs)        

class Selection(torch.nn.Module):

    def __init__(self, classifiers, labels_index, n_features, **kwargs) -> None:
        super().__init__()
        self.index_start = labels_index
        self.n_features = n_features

        # default last_choice to empty tensor
        self.last_choice = torch.tensor([])
        
        self.classifiers = classifiers
        self.classifier0 = classifiers[0]
        self.classifier1 = classifiers[1]
        self.classifier2 = classifiers[2]
        for idx, classifier in enumerate(classifiers):
            self.add_module("classifiers_{}".format(idx) , classifier)

    def parameters(self, recurse: bool = True):
        params = []
        for c in self.classifiers:
            params.extend(c.parameters(recurse=recurse))
        return params

    def get_labels(self, x):
        return x[self.index_start:].view(-1, self.n_features)

    def get_index(self, input):
        index = torch.tensor(data=[None]).to(device=input.device)

        # Unrolled classifiers loop to concatenate to index
        index = torch.concat([index, self.classifier0(input)], dim=1 if input.dim() > 1 else 0)
        index = torch.concat([index, self.classifier1(input)], dim=1 if input.dim() > 1 else 0)
        index = torch.concat([index, self.classifier2(input)], dim=1 if input.dim() > 1 else 0)

        return index

    def choose(self,x, y, bias):
        # No longer using bias when computing last_choice
        self.last_choice = torch.argmax(x, dim=0)

        index = self.last_choice.expand([1, y.shape[1]])
        
        # take the best choice between the given labels
        return torch.gather(y, dim=0, index=index).squeeze()

    # bias defaults to an empty tensor
    def forward(self, x, bias=torch.tensor([])):
        output = self.get_index(x)          
        if not self.training:   
            y = self.get_labels(x)
            output = self.choose(output, y, bias)
        return output
class CategoricalSelection(LinearWithSoftmax):
    
    def __init__(self, n_features, **kwargs) -> None:        
        # need double the number of outputs for mean and std  
        super().__init__(**kwargs)
        self.n_features = n_features
        self.last_choice = None


    def forward(self, x):                

        output = super().forward(x)
                
        if not self.training:
            output = self.choose(output, x)                            
        
        # if training, the choice index probabilities must be made available for sampling
        return output

    def get_labels(self, x):
        return x[self.index_start:].view(-1, self.n_features)

    def get_index(self, input):
        return torch.argmax(input, dim=0)        

    def choose(self, index, x):
        y = self.get_labels(x)
    
        self.last_choice = self.get_index(index)
        index = self.last_choice.expand([1, y.shape[1]])

        # take the best choice between the given labels
        output = torch.gather(y, dim=0, index=index).squeeze()

        return output

