import sys
import time
import torch
from torch.autograd import Variable

torch.backends.cudnn.deterministic = True

# torch.cuda.get_rng_state(device) torch.cuda.set_rng_state(state, device)

from model import layers

def forward_pass():


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        self.output = []
        self.input = []
        for i, layer in enumerate(self.layers):
            # detach from previous history
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)

            # compute output
            x = layer(x)

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
