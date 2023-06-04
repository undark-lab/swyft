""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, s = 1, shallow = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64//s))
        self.down1 = (Down(64//s, 128//s))
        self.down2 = (Down(128//s, 256//s))
        self.down3 = (Down(256//s, 512//s))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512//s, 1024 // s // factor))
        self.up1 = (Up(1024//s, 512//s // factor, bilinear))
        self.up2 = (Up(512//s, 256//s // factor, bilinear))
        self.up3 = (Up(256//s, 128//s // factor, bilinear))
        self.up4 = (Up(128//s, 64//s, bilinear))
        self.outc = (OutConv(64//s, n_classes))
        if shallow:
            self.forward = self.forward_shallow
        else:
            self.forward = self.forward_normal
                

    def forward_shallow(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
#        x4 = self.down3(x3)
#        x5 = self.down4(x4)
#        x = self.up1(x5, x4)
#        x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        
    def forward_normal(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)