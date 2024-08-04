import torch
from torch import nn 
import torch. functional as f


class AlexNet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1)

        self.relu = nn.ReLU()

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=1024, out_features=4096)
        self.fc2= nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=out_classes)

    def forward(self, x):
        # CNN block-1
        x = self.conv1(x)
        x = self.relu(x) # no pooling

        # CNN block-2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)

        # CNN block-3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)

        # CNN block-4
        x = self.conv4(x)
        x = self.relu(x) # no pooling
        
        #CNN block-5
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pooling(x)

        # flattening (conversion from matrices to 1D vector)
        x = x.view(-1, 256*2*2) #flatten the output of the convolution layer and number of channels

        #Firt fully connected layer
        x = self.fc1(x)
        x = self.relu(x) 

        #second fully connected layer
        x = self.fc2(x)
        x = self.relu(x)

        #third FC layer
        x = self.fc3(x)
        x = self.relu(x)

        return x
        

if __name__ == "__main__":
    ## generate random data
    x = torch.rand(size=(8, 3, 224,224))
    print(x.shape)

    ## define model
    model = AlexNet(in_channels=3, out_classes=4)

    ## forward pass
    output = model(x)
    print("output shape: ", output.shape)
    print("end")


                          


    