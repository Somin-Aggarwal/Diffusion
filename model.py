import torch
import torch.nn as nn
from einops import repeat

class SelfAttention(nn.Module):
    def __init__(self, in_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.k = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.v = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        
        # softmax(QK^T/sqrt(d_k))V
        batch_size, channels, height, width = x.shape

        q = torch.reshape(self.q(x),shape=(batch_size, channels, height*width)) # (b,edim,h*w)
        k = torch.reshape(self.k(x),shape=(batch_size, channels, height*width))
        v = torch.reshape(self.v(x),shape=(batch_size, channels, height*width))

        attn_weights = torch.bmm(q.permute(0,2,1), k) / (channels ** 0.5)  # (b,h*w,h*w)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(v, attn_weights.permute(0,2,1))  # (b,edim,h*w)
        attn_output = torch.reshape(attn_output, shape=(batch_size, channels, height, width))
        return attn_output
    
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

class UNET_old(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
    
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.neck1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.neck2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.TransConv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3,padding=1)
        
        self.TransConv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3,padding=1)
        
        self.TransConv3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3,padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        

    def forward(self, x, t):
        
        # Encoder
        features1 = self.conv2(self.conv1(x))
        features2 = self.conv4(self.conv3(self.maxpool(features1)))
        features3 = self.conv6(self.conv5(self.maxpool(features2)))
        
        # Neck
        final_features = self.neck2(self.neck1(self.maxpool(features3)))
        
        t = t.view(-1, 1).float()                  
        t_emb = self.time_mlp(t)                   
        
        t_emb = t_emb[:, :, None, None]                   
        final_features = final_features + t_emb    
        
        output = self.TransConv1(final_features)
        output = torch.concatenate([output, features3], dim=1)
        output = self.conv8(self.conv7(output))
        
        output = self.TransConv2(output)
        output = torch.concatenate([output, features2], dim=1)
        output = self.conv10(self.conv9(output))
        
        output = self.TransConv3(output)
        output = torch.concatenate([output, features1], dim=1)
        output = self.conv12(self.conv11(output))
        
        return output

      
if __name__=="__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(image_channels=1,time_dim=128).to(device)
    dummy_input = torch.randn(size=(1,1,32,32),device=device)
    time = torch.randn(size=(1,1),device=device)
    y = torch.tensor([0,],device=device)
    
    output = model(dummy_input, time)
    print(output.shape)
    
    param_count = 0
    for params in model.parameters():
        param_count += params.numel()  
    print(param_count)    
    # print(model)