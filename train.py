import time

import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import transforms as t

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from matplotlib import pyplot as plt 


## custom import 
from dataset import MyDataset
from model import secondModel




def train_one_epoch(model, train_dl, test_dl, optimizer, loss_fn, device, epoch):
   
    total_loss = 0
    total_train_samples = 0
    total_correct_predictions = 0

    ## train model
    model.train()
    ## load data 
    for samples, labels in  train_dl:
        samples = samples.to(device) #??? cpu or gpu?? what is x?, predicted or target?
        labels = labels.to(device) # y???

        ## forward pass
        predictions = model(samples)

        ## loss calculation
        loss = loss_fn(predictions, labels) #loss_fn is argument??? and prediction, y???

        ## backward pass
        optimizer.zero_grad() # cache ----> clear / zero
        loss.backward()
        optimizer.step()

        total_loss += loss
         ## model accuracy
        out_probs = torch.softmax(predictions, dim=1) #
        _, predicted = torch.max(out_probs.data, 1)

        total_train_samples += labels.size(0) # this formula is set manually. we can also take from internet
        total_correct_predictions += (predicted == labels).sum().item()

    mean_loss =  total_loss/len(train_dl)
    total_accuracy = (total_correct_predictions / total_train_samples) * 100.0

    return total_accuracy, mean_loss

def evaluate_model(model, test_dl, loss_fn, device, epoch):
    total_loss = 0

    total_samples = 0
    total_correct_predictions = 0 # this variable name is same like in train mode.

    model.eval()
    with torch.no_grad():
        for x, y in test_dl: # x,y means samples and labels???
            x, y = x.to(device), y.to(device)

            prediction = model(x)
            loss = loss_fn(prediction, y) #y??

            total_loss += loss #??? = total_loss + loss

            ## model accuracy
            out_probs = torch.softmax(prediction, dim=1) #
            _, predicted = torch.max(out_probs.data, 1) # _ ??? and what is .data and 1???

            total_samples += y.size(0) # this formula is set manually. we can also take from internet
            total_correct_predictions += (predicted == y).sum().item() #????

    total_accuracy = (total_correct_predictions / total_samples) * 100.0   
    mean_loss = total_loss / len(test_dl)

    return total_accuracy, mean_loss


if __name__ == "__main__":
    data_dir = 'Images'
    data_type_train = 'train'
    data_type_test = 'test'
    batch_size = 256
    epochs = 10
    in_channels = 3
    output_classes = 4
    learning_rate = 0.001
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transforms = t.Compose([t.Resize((224, 224)),
                            t.ToTensor()])
    ## train and test dataset
    dataset_train = MyDataset(data_dir, data_type_train, transforms)
    dataset_test = MyDataset(data_dir, data_type_test, transforms)

    ## test and train data loader
    train_dl = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    ## define model, loss_fn and optimizer
    model = secondModel(in_channels=in_channels, out_classes=output_classes).to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)


    ## model training and testing 
    for epoch in range(epochs):
        train_accuracy, train_loss = train_one_epoch(model, train_dl, test_dl, optimizer, loss_fn, device, epoch)
        test_accuracy, test_loss = evaluate_model(model, test_dl, loss_fn, device, epoch)
        
        print(f'Epoch {epoch} completed')
        print(f'train loss of the model after epoch {epoch}: {train_loss}')
        print(f'test loss of the model after epoch {epoch}: {test_loss}')
        print(f'train accuracy of the model after epoch {epoch}: {train_accuracy}')
        print(f'test accuracy of the model after epoch {epoch}: {test_accuracy}')
        print("*******************************\n\n")