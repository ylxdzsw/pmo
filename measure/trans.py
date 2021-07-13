import time
import torch
import torch.nn as nn
from torch.autograd import Variable

import sys

batchsize = int(sys.argv[1])
seqlen = 32
emsize = 2048 # embedding dimension
nhid = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 16 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

records = {}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([ nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers) ])

    def forward(self, x):
        self.output = []
        self.input = []
        for i, layer in enumerate(self.layers):
            # detach from previous history
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)

            # compute output
            torch.cuda.current_stream().synchronize()
            tic = time.time()

            x = layer(x)

            torch.cuda.current_stream().synchronize()
            toc = time.time()

            records[f"f{i}"] = toc - tic

            # add to list of outputs
            self.output.append(x)
        return x

    def backward(self, g):
        for i, output in reversed(list(enumerate(self.output))):
            torch.cuda.current_stream().synchronize()
            tic = time.time()

            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
            else:
                output.backward(self.input[i+1].grad.data)

            torch.cuda.current_stream().synchronize()
            toc = time.time()

            records[f"b{i}"] = toc - tic

def show_size(layers):
    for i, layer in enumerate(layers):
        bsize = sum( p.nelement() * p.element_size() for p in layers.parameters() )
        print(f"l{i}", bsize >> 20)
        print(f"n{i}", (bsize >> 20) / 46 / 1000)
        print(f"p{i}", (bsize >> 20) / 4.8 / 1000)

if __name__ == '__main__':
    model = Net()
    inp = Variable(torch.randn(seqlen, batchsize, emsize))
    output = model(inp)
    gradients = torch.randn(*output.size())
    model.backward(gradients)

    for i in range(nlayers):
        print(f"f{i}", records[f"f{i}"])

    for i in range(nlayers):
        print(f"b{i}", records[f"b{i}"])

    show_size(model.layers)
