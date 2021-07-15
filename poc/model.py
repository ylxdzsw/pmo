import torch.nn as nn

batchsize = 64
seqlen = 32
emsize = 2048 # embedding dimension
nhid = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 8 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 16 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

layers = [ nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers) ]
