import torch
import torch.nn as nn


class FCLSTM(nn.Module):
    def __init__(self,inputNode=2,hiddenNode = 128, outputNode=1):   
        super(FCLSTM, self).__init__()
             
        self.inp = nn.Linear(inputNode, hiddenNode)
        self.rnn = nn.LSTM(hiddenNode , hiddenNode , 2) #input,hidden,layers
        #self.rnn = nn.RNN(hiddenNode , hiddenNode , 3)
        self.out = nn.Linear(hiddenNode, 1)
        
        self.outputNode = outputNode
        
    def step(self,input,hidden = None):

        input = self.inp(input).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output,hidden
    
    def forward(self, inputs,steps=40, hidden=None):
        
        #format: batch, seq, features
        outputs = torch.zeros(inputs.shape[0], inputs.shape[1], self.outputNode)
        
        for i in range(steps):
            input = inputs[:,i]
            
            #print(input.shape)
            output, hidden = self.step(input, hidden)
            
            #print(output.shape)
            outputs[:,i] = output
        
        return outputs