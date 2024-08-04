import torch
from torch import nn 
import torch. functional as f


class secondModel(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(in_features=256*14*14, out_features=1024) 
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=out_classes)


    def forward(self, x): ###x????
        ## First CNN block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)

        ## second cnn block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)

        ## 3rd CNN block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)

        ## fourth CNN block
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pooling(x)

        # Now flattening (conversion from matrices to vector)
        x = x.view(-1, 256*14*14) # flatten the output (50176) of the final convolution layer (size of image = (14, 14), No of. channels = (256))
        # First fully connected (fc) layer
        x = self.relu(self.fc1(x))

        ## 2nd fc layer
        x = self.relu(self.fc2(x))

        ## final layer 

        x = self.relu(self.fc3(x))

        return x
    


# if __name__ == "__main__":
#     ## generate random data
#     x = torch.rand(size=(8, 3, 224,224))
#     print(x.shape)

#     ## define the model
#     model = secondModel(in_channels=3, out_classes=4)

#     ## forward pass
#     output = model(x)
#     print("output shape: ", output.shape)
#     print("end")

















