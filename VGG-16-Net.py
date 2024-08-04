import torch
from torch import nn 
import torch.functional as f
import numpy as np

class VGG16(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()

        self.in_channels = in_channels
        self.out_classess = out_classes

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv9 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv10 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv11 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv12 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv13 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)

        self.relu = nn.ReLU()

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=out_classes)

    def forward(self, x):
        ## first CNN block
        x = self.conv1(x)
        x = self.relu(x)
        ## 2nd CNN block
        x = self.conv2(x) 
        x = self.relu(x)
        x = self.pooling(x)
        ## 3rd CNN block
        x = self.conv3(x)
        x = self.relu(x)
        ## 4th CNN block
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pooling(x)

        #5th CNN block
        x = self.conv5(x)
        x = self.relu(x)

        #6th CNN block
        x = self.conv6(x)
        x = self.relu(x)

        #7th CNN block
        x = self.conv7(x)
        x = self.relu(x)
        x = self.pooling(x)

        #8th CNN block
        x = self.conv8(x)
        x = self.relu(x)
        

        #9th CNN block
        x = self.conv9(x)
        x = self.relu(x)

        #10th CNN block
        x = self.conv10(x)
        x = self.relu(x)
        x = self.pooling(x)

        #11th CNN block
        x = self.conv11(x)
        x = self.relu(x)
        
        #12th CNN block
        x = self.conv12(x)
        x = self.relu(x)

        #13th block
        x = self.conv13(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = x.view(-1, 512*7*7)

        # first fc layer
        x = self.fc1(x)
        x = self.relu(x)

        # 2nd fc layer
        x = self.fc2(x)
        x= self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return x

def get_n_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params
 
if __name__ == "__main__":
    ## generate random data 
    x = torch.rand(size=(8, 3, 224,224))
    print(x.shape)

    ## define model
    model = VGG16(in_channels=3, out_classes=4)

    ## forward pass
    output = model(x)
  
    model.eval()
    with torch.no_grad():
        x = model(x)



    

















































