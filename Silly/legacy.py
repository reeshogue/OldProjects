class Old_Block(nn.Module):
    def __init__(self, sample_size, inchan, outchan, kernel_size=7):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(inchan, inchan, kernel_size)
        self.conv2 = nn.Conv2d(inchan, outchan, kernel_size)
        self.conv_shortcut = nn.Conv2d(inchan, outchan, kernel_size)
        self.conv3_mu = nn.Conv2d(outchan, outchan, kernel_size)
        self.conv3_stddev = nn.Conv2d(outchan, outchan, kernel_size)
        self.conv3_forget = nn.Conv2d(outchan, outchan, kernel_size)
        self.sample = nn.Upsample((sample_size, sample_size))

    def forward(self, x):
        y = self.sample(self.conv2(self.conv(x)))

        y_stddev = self.sample(self.conv3_stddev(y))
        y_mu = self.sample(self.conv3_mu(y))

        x = self.sample(self.conv_shortcut(x))
        
        x_forget = self.sample(torch.sigmoid(self.conv3_forget(x)))

        y = (y + x) #+ (x_forget * x) # + (y_stddev * torch.randn_like(y) + y_mu)
        return y

class OldRees_Block(nn.Module):
    def __init__(self, sample_size, inchan, outchan, kernel_size=5, exp_fact=2, rees=2):
        super(OldRees_Block, self).__init__()
        midchan = inchan * exp_fact
        self.rees = rees
        self.pointwise = nn.Conv2d(inchan, midchan, 1)
        self.bn1 = nn.BatchNorm2d(midchan)

        self.depthwise_list = nn.ModuleList([nn.Conv2d(midchan, midchan, kernel_size, padding=kernel_size//2,
            groups=midchan) for i in range(rees)])
        self.channel_shuffle = ChannelShuffle()

        self.linear = nn.Conv2d(midchan, outchan, 1)
        
        
        self.shortcut = nn.Conv2d(inchan, outchan, 1)
        self.sample = nn.Upsample((sample_size, sample_size))
        self.activation = Rish()

    def forward(self, inp):
        y = self.activation(self.bn1(self.pointwise(inp)))
        rees = self.depthwise_list[0](y)
        for i in range(1, self.rees):
            rees_y = self.depthwise_list[i](y)
            rees += rees_y
        rees /= self.rees
        y = rees
        
        y = self.linear(y)
        y = self.sample(y)
        
        print(y.shape)
        
        shortcut = self.shortcut(inp)
        shortcut = self.sample(shortcut)
        return y + shortcut
