import torch.nn as nn
import torch

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes,dropout_rate,hidden_size):
        super(LogisticRegressionModel, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(input_size + 1, hidden_size)
        self.activation=nn.ReLU()
        self.linear2 =nn.Linear(hidden_size,num_classes)
        self.linear = nn.Linear(input_size+1, num_classes)

    def forward(self, x,hidden_size):
        
        x_new = torch.cat((x,torch.ones(x.shape[0],1)),1)
        x_new = self.dropout(x_new) 
        if hidden_size != 0:
            x_new = self.activation(self.linear1(x_new))
            out = self.linear2(x_new)
        else:
            out = self.linear(x_new)
        return out
