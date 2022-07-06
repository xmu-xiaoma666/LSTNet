from numpy.core.shape_base import stack
from torch.nn import functional as F
from torch.nn.modules.activation import GELU
from .utils import PositionWiseFeedForward
import torch
from torch import nn
from .attention import MultiHeadAttention
from ..relative_embedding import GridRelationalEmbedding
from .repblobk import LocalPerceptron
import numpy as np
import math


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)
        self.lp=LocalPerceptron(d_model)

    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None, pos=None):
        att1 = self.mhatt(queries, keys, values, relative_pos, attention_mask, attention_weights)
        att1= self.lnorm(queries + self.dropout(att1))

        att2 = self.lp(att1)
        att2= self.lnorm2(att1 + self.dropout2(att1))
        
        ff = self.pwff(att2)

        return ff


def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x
def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=3):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,n,c=x_all.shape
        a=torch.sum(torch.sum(x_all,1),1) #bs,c
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) #bs,kc
        hat_a=hat_a.reshape(b,self.k,c) #bs,k,c
        bar_a=self.softmax(hat_a) #bs,k,c
        attention=bar_a.unsqueeze(-2) # #bs,k,1,c
        out=attention*x_all # #bs,k,n,c
        out=torch.sum(out,1)
        return out



class S2Fuse(nn.Module):
    def __init__(self, channels,k):
        super().__init__()
        self.k=k
        self.channels=channels
        self.split_attention = SplitAttention(channels,k)
        self.mlp=nn.Sequential(
            nn.Conv2d(k*channels,k*channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(k*channels),
            nn.ReLU(),
            nn.Conv2d(k*channels,channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self,x):
        b,n,c_big = x.size() #bs,n,dim*3
        h,w=int(math.sqrt(n)),int(math.sqrt(n))
        x=x.reshape(b,h,w,c_big)
        x1 = spatial_shift1(x[:,:,:,:c_big//3]) #bs,h,w,dim
        x2 = spatial_shift2(x[:,:,:,c_big//3:c_big//3*2]) #bs,h,w,dim
        x3 = x[:,:,:,c_big//3*2:] #bs,h,w,dim
        x_all=torch.cat([x1,x2,x3],-1) #bs,h,w,3*dim
        out = self.mlp(x_all.permute(0,3,1,2)).permute(0,2,3,1) #bs,h,w,dim
        out=out.reshape(b,-1,self.channels) #bs,n,dim
        # print(out.shape)
        return out

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])
        self.s2fuse=S2Fuse(d_model,3)

    def forward(self, input, attention_weights=None, pos=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        relative_geometry_embeddings = GridRelationalEmbedding(input.shape[0])
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [w(flatten_relative_geometry_embeddings).view(box_size_per_head) for w in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        grid2grid = F.relu(relative_geometry_weights)
        outs=[]
        out = input #bs,n,dim
        for l in self.layers:
            out = l(out, out, out, grid2grid, attention_mask, attention_weights, pos=pos)
            outs.append(out)
        stack_out=torch.cat(outs,-1) #bs,n,dim*3
        stack_out=self.s2fuse(stack_out)
        out=out+0.2*stack_out

        return out, attention_mask


from numpy import math
import numpy as np
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        rowPE = torch.zeros(max_len,max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        rowPE[ :,:,0::2] = torch.sin(position * div_term)
        rowPE[ :,:, 1::2] = torch.cos(position * div_term)
        colPE=rowPE.transpose(1, 0)
        rowPE = rowPE.unsqueeze(0)
        colPE = colPE.unsqueeze(0)
        self.rowPE=rowPE.cuda()
        self.colPE=colPE.cuda()

    def forward(self, x):
        feat=x
        bs,gs,dim=feat.shape
        feat=feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
        feat = feat + self.rowPE[:, :int(np.sqrt(gs)), :int(np.sqrt(gs)),  :dim ]+ self.colPE[:,  :int(np.sqrt(gs)),  :int(np.sqrt(gs)),  :dim ]
        feat=feat.view(bs,-1,dim)
        return self.dropout(feat)


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pe=PositionalEncoding(d_model=d_in,dropout=0)

    def forward(self, input, attention_weights=None):
        feat=self.pe(input)
        mask = (torch.sum(feat, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(feat))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)
