import torch
import torch.nn as nn
from einops import repeat
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class SelfAttentionDiff(nn.Module):
    def __init__(self, in_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_ch = in_ch

        self.group_norm = nn.GroupNorm(num_groups=min(in_ch,32), num_channels=in_ch)

        
        self.q = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.k = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.v = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        
        self.proj_out = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        
        # softmax(QK^T/sqrt(d_k))V
        b, c, h, w = x.shape
        
        x_dash = self.group_norm(x)

        q = self.q(x_dash).view(b, c, -1) ##
        k = self.k(x_dash).view(b, c, -1) # (b,c,h*w)
        v = self.v(x_dash).view(b, c, -1) ##
        
        attn_weights = torch.bmm(q.permute(0,2,1), k) / (c ** 0.5)  # (b,h*w,h*w)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.bmm(v,attn_weights.permute(0,2,1))  # (b,h*w,edim)
        
        out = out.view(b, c, h, w) 
        out = self.proj_out(out)
               
        return x + self.gamma * out
    
class ResBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, tdim=None):
        super(ResBlock_2, self).__init__()
        self.gn1 = nn.GroupNorm(num_groups=min(in_channels,32), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.gn2 = nn.GroupNorm(num_groups=min(out_channels,32), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        self.silu = nn.SiLU()
        
        self.tdim = tdim
        if tdim is not None:
            self.time_linear = nn.Linear(tdim,out_channels)
        
    def forward(self, x, t):
        
        identity = self.shortcut(x)
        
        out = self.gn1(x)
        out = self.silu(out)
        out = self.conv1(out)
        
        if self.tdim is not None:
            t_emb = self.time_linear(t)
            t_emb = repeat(t_emb, 'b e -> b e h w', h=out.shape[2], w=out.shape[3])
            out = out + t_emb

        out = self.gn2(out)
        out = self.silu(out)
        out = self.conv2(out)

        out = out + identity
        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(out_channels,32), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(out_channels,32), num_channels=out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()
        
        self.tdim = tdim
        if tdim is not None:
            self.time_linear = nn.Linear(tdim,out_channels)
        
    def forward(self, x, t):
        
        identity = self.shortcut(x)
        out = self.relu(self.gn1(self.conv1(x)))
        if self.tdim is not None:
            t_emb = self.time_linear(t)
            t_emb = repeat(t_emb, 'b e -> b e h w', h=out.shape[2], w=out.shape[3])
            out = out + t_emb
        out = self.gn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out

class UNET(nn.Module):
    def __init__(self, image_channels, time_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.image_channels = image_channels
        
        self.block1 = ResBlock(in_channels=image_channels, out_channels=8, tdim=time_dim)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.block2 = ResBlock(in_channels=8, out_channels=32, tdim=time_dim)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.block3 = ResBlock(in_channels=32, out_channels=128, tdim=time_dim)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)   

        self.neck1 = ResBlock(in_channels=128, out_channels=256, tdim=time_dim)  
        self.neck2 = ResBlock(in_channels=256, out_channels=128, tdim=time_dim)

        self.TransConv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.Upblock1 = nn.Sequential(ResBlock(in_channels=256, out_channels=128, tdim=time_dim),
                                       ResBlock(in_channels=128, out_channels=32, tdim=time_dim)
                                       )
        
        self.TransConv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.Upblock2 = nn.Sequential(ResBlock(in_channels=64, out_channels=32, tdim=time_dim),
                                       ResBlock(in_channels=32, out_channels=8, tdim=time_dim)
                                       )
        
        self.TransConv3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)
        self.Upblock3 = nn.Sequential(ResBlock(in_channels=16, out_channels=16, tdim=time_dim))
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=image_channels, kernel_size=1) 
        
        self.tdim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.num_classes = 10
        
    def forward(self, x, t):
        
        t = t.view(-1, 1).float()                  
        t_emb = self.time_mlp(t)       
                    
        # Encoder
        features1 = self.block1(x,t_emb) # 32x32x8
        features2 = self.block2(self.maxpool1(features1), t_emb) # 16x16x32
        features3 = self.block3(self.maxpool2(features2), t_emb) # 8x8x128
        
        # Neck
        final_features = self.neck2(self.neck1(self.maxpool3(features3), t_emb),t_emb)
                
        final_features = final_features + t_emb[:, :, None, None]    
        
        output = self.TransConv1(final_features)
        output = torch.concatenate([output, features3], dim=1)
        for block in self.Upblock1:
            output = block(output,t_emb)
        
        output = self.TransConv2(output)
        output = torch.concatenate([output, features2], dim=1)
        for block in self.Upblock2:
            output = block(output,t_emb)
        
        output = self.TransConv3(output)
        output = torch.concatenate([output, features1], dim=1)
        for block in self.Upblock3:
            output = block(output,t_emb)
        output = self.final_conv(output)
        
        return output

class UNET_Cifar10(nn.Module):
    def __init__(self, image_channels:int, channels:list, attn_bool:list, n:int, time_dim:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        '''
        n : Number of resblocks per resolution
        '''
        
        assert len(attn_bool) == len(channels) - 1
        self.image_channels = image_channels
        
        self.proj_in = nn.Conv2d(in_channels=image_channels, out_channels=channels[0], kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # if you use nn.module list then it send the layers and tensors to device 
        # but if you use a normal list it will not detect the layers
        
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            block = nn.ModuleList()
            block.append(ResBlock_2(in_channels=channels[i],out_channels=channels[i+1], tdim=time_dim))
            for j in range(n-1):
                block.append(ResBlock_2(in_channels=channels[i+1],out_channels=channels[i+1], tdim=time_dim))
            if attn_bool[i]:
                block.append(SelfAttentionDiff(in_ch=channels[i+1]))
            block.append(nn.Conv2d(in_channels=channels[i+1],out_channels=channels[i+1],kernel_size=3,stride=2,padding=1))
            self.encoder_blocks.append(block)
        
        self.neck1 = ResBlock_2(in_channels=channels[-1], out_channels=channels[-1], tdim=time_dim)  
        self.attn_neck = SelfAttentionDiff(in_ch=channels[-1])
        self.neck2 = ResBlock_2(in_channels=channels[-1], out_channels=channels[-1], tdim=time_dim)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels)-1,0,-1):
            block = nn.ModuleList()
            block.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1))
            )
            block.append(ResBlock_2(in_channels=2*channels[i],out_channels=channels[i-1],tdim=time_dim))
            for j in range(n-1):
                block.append(ResBlock_2(in_channels=channels[i-1],out_channels=channels[i-1],tdim=time_dim))
            if attn_bool[i-1]:
                block.append(SelfAttentionDiff(in_ch=channels[i-1]))
            self.decoder_blocks.append(block)
                
        
        self.tdim = time_dim
        self.get_time_emb = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )
        
        self.final_norm = nn.GroupNorm(num_groups=min(32,channels[0]), num_channels=channels[0])
        self.final_act = nn.SiLU()
        self.proj_out = nn.Conv2d(channels[0], image_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        
        t_emb = self.get_time_emb(t)
        t_emb = t_emb.squeeze(0)
        t_emb = self.time_mlp(t_emb)    
        
        out = self.proj_in(x)
        
        features = []
        for block in self.encoder_blocks:
            for layer in block[:-1]:
                if isinstance(layer, ResBlock_2):
                    out = layer(out, t_emb)
                else:
                    out = layer(out)
            features.append(out)
            out = block[-1](out)

        out = self.neck1(out,t_emb)
        out = self.attn_neck(out)
        out = self.neck2(out,t_emb)
        
        n_ft = len(features) 
                
        for i,block in enumerate(self.decoder_blocks):
            out = block[0](out)
            out = torch.concatenate((out,features[n_ft - 1 - i]),dim=1)
            for layer in block[1:]:
                if isinstance(layer, ResBlock_2):
                    out = layer(out, t_emb)
                else:
                    out = layer(out)

        out = self.final_norm(out)
        out = self.final_act(out)
        out = self.proj_out(out)
        
        return out
     
if __name__=="__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET_Cifar10(
        image_channels=3,         # RGB images
        channels=[64, 128, 256, 512],  # Base width 64. Bottleneck will be at 8x8.
        attn_bool=[False, True, True],   # Attention at 16x16 and 8x8 resolutions.
        n=2,                      # 2 ResBlocks per level (Industry Standard)
        time_dim=256              # Typically 4x the base channel size (4 * 64)
    ).to(device)
    dummy_input = torch.randn(size=(2,3,32,32),device=device)
    time = torch.randn(size=(1,1),device=device)
       
    output = model(dummy_input, time)
    print(output.shape)
    
    param_count = 0
    for params in model.parameters():
        param_count += params.numel()  
    print(param_count)    
