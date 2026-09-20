import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class PID_LAYER(nn.Module):
    def __init__(self, regulation_heads, input_dim, output_dim, cycle):
        super().__init__()
        dimensions=5
        self.cycle = cycle
        self.hyperparameters_input_layer = nn.Linear(input_dim, regulation_heads*dimensions)
        '''self.adition_vector = nn.Sequential(nn.Linear(input_dim,1), nn.Tanh())'''
        self.regulation_head = nn.Linear(regulation_heads, input_dim)
        self.mid_layer_hyper_parametres = nn.Sequential(nn.Linear(regulation_heads, regulation_heads*dimensions)
                                                        , nn.Linear(regulation_heads*dimensions,regulation_heads*dimensions)
                                                        , nn.Linear(regulation_heads*dimensions,regulation_heads*dimensions)
                                                        , nn.Linear(regulation_heads*dimensions,regulation_heads*dimensions))
        self.token_head = nn.Linear(input_dim, output_dim)

        self.regulation_heads = regulation_heads
        self.dimensions = dimensions
        self.input_dim = input_dim
        
    def forward(self, x):
        slices = self.dimensions

        e_previous = 0.00
        I_out = 0.00

        for _ in range(self.cycle):

            if _ == 0:
                _in = self.hyperparameters_input_layer.forward(x)
            else:
                _in = self.mid_layer_hyper_parametres.forward(regulation_output)
            
            hyperparameters = torch.chunk(_in, slices)
            Ki = (hyperparameters[0])
            Kp = (hyperparameters[1])
            Kd = (hyperparameters[2])
            s = hyperparameters[3]
            e = hyperparameters[4]

            P = Kp @ e
            
            P = P.repeat(self.regulation_heads)

            I_out = I_out + e

            D_out = e - e_previous

            regulation_output = P + Ki*I_out + Kd*D_out + s
            
            self.e_previous = e
        return x
    
    
    
    def state_to_token(self, x):
        x = x.unsqueeze(1).T
        x = self.token_head.forward(x)
        return x
    