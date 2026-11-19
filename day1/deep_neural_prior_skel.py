from torch.optim import Adam
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms 
from matplotlib import pyplot as plt


class MyUNet(nn.Module):
    def __init__(self):
        super(MyUNet, self).__init__()
        
        #Down layer 1
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #Up layers:
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=)
        )


        # TODO


    def forward(self,x):

        #TODO

        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

myunet = MyUNet()
myunet.to(device)


input_img = Image.open('testcrop.jpg')
transform = transforms.Compose([transforms.PILToTensor()])
target = transform(input_img)/255
target = target.unsqueeze(0).float().to(device)

h = target.size()[2]
w = target.size()[3]
z = torch.rand(1,16, h, w).to(device)


#Loss and optimizer
criterion = 
optimizer = Adam()
losslog = []

#test
for i in range(0,2000):

    ##TODO

output = myunet(z)

plt.imsave('final.jpg', output[0].cpu().detach().permute(1,2,0).numpy())

# display the loss
#plt.figure(figsize=(6,4))
#plt.yscale('log')
#plt.plot(losslog, label = 'loss ({:.4f})'.format(losslog[-1]))
#plt.xlabel("Epochs")
#plt.legend()
#plt.show()
#plt.close()
