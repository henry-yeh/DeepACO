import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, 
                 ntoken_input = 6,
                 d_model = 32, 
                 nhead = 2, 
                 d_hid = 32,
                 nlayers = 3,
                 dropout = 0
                 ):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken_input, d_model)
        self.d_model = d_model
        self.decoder_heu = ParNet()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        heu = self.decoder_heu(output).squeeze()
        heu = heu / heu.max()
        return heu


class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x) # last layer
        return x

# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='relu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)
    def forward(self, x):
        return super().forward(x).squeeze(dim = -1)
    

if __name__ == '__main__':
    from utils import gen_instance
    m = 5
    price, weight = gen_instance(m=m)
    src = torch.cat((price.T.unsqueeze(-1), weight.T), dim=-1)
    src.unsqueeze_(1)
    print(src.shape)
    
    net = TransformerModel(ntoken_input=m+1)
    phe, heu = net(src)
    print(phe, heu)