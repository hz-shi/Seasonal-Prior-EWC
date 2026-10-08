import torch
import math
import random
from torch import nn
from src.model.iam4vp.modules import ConvSC, ConvNeXt_block, Learnable_Filter, Attention, ConvNeXt_bottle

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim,) * -emb).to(x.device)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim*4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim*4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1

class LP(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(LP,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, seq_len):
        super(Decoder,self).__init__()
        self.seq_len = int(seq_len)
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid * self.seq_len, 64, 1)


    def forward(self, hid, enc1=None, batch_size=None, seq_len=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        if batch_size is None:
            raise ValueError("Decoder.forward requires batch_size.")
        seq_len = self.seq_len if seq_len is None else int(seq_len)
        if seq_len != self.seq_len:
            raise ValueError(f"Decoder seq_len mismatch: expected {self.seq_len}, got {seq_len}.")
        b_t, channels, height, width = Y.shape
        expected = int(batch_size) * self.seq_len
        if b_t != expected:
            raise ValueError(f"Decoder temporal shape mismatch: got {b_t}, expected {expected}.")
        Y = Y.reshape(int(batch_size), self.seq_len * channels, height, width)
        Y = self.readout(Y)
        return Y

class Predictor(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T):
        super(Predictor, self).__init__()

        self.N_T = N_T
        st_block = [ConvNeXt_bottle(dim=channel_in)]
        for _ in range(0, N_T):
            st_block.append(ConvNeXt_block(dim=channel_in))

        self.st_block = nn.Sequential(*st_block)

    def forward(self, x, time_emb):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        z = self.st_block[0](x, time_emb)
        for i in range(1, len(self.st_block)):
            z = self.st_block[i](z, time_emb)

        y = z.reshape(B, int(T/2), C, H, W)
        return y


def _encoded_spatial_dim(size, n_s):
    out = int(size)
    for stride in stride_generator(n_s):
        if stride == 2:
            out = (out + 1) // 2
    return out

class IAM4VP(nn.Module):
    def __init__(self, shape_in, hid_S=64, hid_T=512, N_S=4, N_T=6):
        super(IAM4VP, self).__init__()
        T, C, H, W = shape_in
        self.seq_len = int(T)
        self.input_hw = (int(H), int(W))
        latent_h = _encoded_spatial_dim(H, N_S)
        latent_w = _encoded_spatial_dim(W, N_S)
        self.time_mlp = Time_MLP(dim=64)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Predictor(T*hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S, seq_len=T)
        self.attn = Attention(64)
        self.readout = nn.Conv2d(64, 1, 1)
        self.mask_token = nn.Parameter(torch.zeros(T, hid_S, latent_h, latent_w))
        self.lp = LP(1, hid_S, N_S)

    def forward(self, x_raw, y_raw=None, t=None):
        y_raw = [] if y_raw is None else y_raw
        B, T, C, H, W = x_raw.shape
        if T != self.seq_len:
            raise ValueError(f"IAM4VP expected input length {self.seq_len}, got {T}.")
        if (H, W) != self.input_hw:
            raise ValueError(f"IAM4VP expected spatial size {self.input_hw}, got {(H, W)}.")
        if len(y_raw) > self.seq_len:
            raise ValueError(f"IAM4VP received {len(y_raw)} autoregressive frames, max is {self.seq_len}.")
        if t is None:
            t = torch.zeros(B, device=x_raw.device)
        elif t.ndim == 0:
            t = t.repeat(B)
        t = t.to(device=x_raw.device, dtype=x_raw.dtype).view(B)
        x = x_raw.view(B*T, C, H, W)
        time_emb = self.time_mlp(t)
        embed, skip = self.enc(x)
        mask_token = self.mask_token.repeat(B,1,1,1,1)

        for idx, pred in enumerate(y_raw):
            embed2,_ = self.lp(pred)
            mask_token[:,idx,:,:,:] = embed2

        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        z2 = mask_token
        z = torch.cat([z, z2], dim=1)
        hid = self.hid(z, time_emb)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip, batch_size=B, seq_len=T)
        Y = self.attn(Y)
        Y = self.readout(Y)
        return Y
    
if __name__ == "__main__":
    import numpy as np
    model = IAM4VP([10,1,64,64])
    inputs = torch.randn(2,10, 1,64,64)
    inputs2 = torch.randn(2,10, 1,64,64)
    pred_list = []
    for timestep in range(10):
        t = torch.tensor(timestep*100).repeat(inputs.shape[0])
        out = model(inputs, y_raw=pred_list, t=t)
        pred_list.append(out)
