'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch

def swish(x):
    return x*torch.nn.functional.sigmoid(x)

class CNNBlock(torch.nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(channels//4, channels)

        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=kernel, padding="same")
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=kernel, padding="same")
        self.drop = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        y = swish(self.norm1(x))
        y = swish(self.conv1(y))
        y = swish(self.conv2(y))
        y = self.drop(y)
        y = y + x
        return y


class CNNEncoder(torch.nn.Module):

    def __init__(self, starting_width, bottleneck, device_reference, channels=64, repeat=1):
        super().__init__()
        K = channels
        w = starting_width
        self.device = device_reference

        # encoder --------------------------------------------------------
        
        self.enc_pre_conv = torch.nn.Conv2d(3, K, kernel_size=3, padding="same")
        self.pool = torch.nn.MaxPool2d(2)

        self.enc_blocks = torch.nn.ModuleList()

        for k in range(3):
            for i in range(repeat):
                self.enc_blocks.append(CNNBlock(K))
            self.enc_blocks.append(self.pool)
            w = int(w/2)

        self.enc_post_conv = torch.nn.Conv2d(K, 16, kernel_size=3, padding="same")
        self.flat_size = int(16*w*w)
        self.shaped_size = (16,int(w),int(w))
        self.middle1 = torch.nn.Linear(self.flat_size, bottleneck*2)
        self.middle2 = torch.nn.Linear(bottleneck*2, bottleneck)

        # decoder --------------------------------------------------------
        self.middle_inv1 = torch.nn.Linear(bottleneck, bottleneck*2)
        self.middle_inv2 = torch.nn.Linear(bottleneck*2, self.flat_size)

        self.dec_pre_conv = torch.nn.Conv2d(16, K, kernel_size=3, padding="same")
        self.upsampler = torch.nn.Upsample(scale_factor=2) # 12 -> 24

        self.dec_blocks = torch.nn.ModuleList()

        for k in range(3):
            for i in range(repeat):
                self.dec_blocks.append(CNNBlock(K))
            self.dec_blocks.append(self.upsampler)
            w = int(w*2)

        self.dec_post_block = CNNBlock(K)
        self.dec_post_conv = torch.nn.Conv2d(K, 3, kernel_size=3, padding="same")

        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Encoder / Decoder has", params, "parameters.")


    def encode(self, o):
        o = swish(self.enc_pre_conv(o))
        for mod in self.enc_blocks:
            o = mod(o)

        o = swish(self.enc_post_conv(o))
        if len(o.shape)==3:
            o = torch.flatten(o, start_dim=0)
        else:
            o = torch.flatten(o, start_dim=1)
        o = swish(self.middle1(o))
        o = self.middle2(o)
        return o

    def decode(self, z):
        z = swish(self.middle_inv1(z))
        z = swish(self.middle_inv2(z))
        if len(z.shape)==1:
            z = torch.reshape(z, self.shaped_size)
        else:
            z = torch.reshape(z, (-1, *self.shaped_size))
        z = swish(self.dec_pre_conv(z))

        for mod in self.dec_blocks:
            z = mod(z)

        z = self.dec_post_block(z)
        z = self.dec_post_conv(z)
        return z
