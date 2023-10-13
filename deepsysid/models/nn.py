import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self,inputNode=2,hiddenNode = 256, outputNode=1):   
        super(FC, self).__init__()     
        #Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        
        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, X):
        
        out1 = self.Linear1(X) 
        out2 = self.activation(out1)
        out3 = self.Linear2(out2)
        return out3
    

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