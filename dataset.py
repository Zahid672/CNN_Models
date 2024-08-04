import os 

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as t
from torch.utils.data import Dataset

__all__ = ['MyDataset', 'plot_grid_images']

class MyDataset(Dataset):
    def __init__(self, data_dir, data_type, transform=None, target_transform=None):
        super().__init__()
        self.data_dir = data_dir 
        self.data_type = data_type

        self.image_names, self.labels = self.__process_data()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]

        label_map = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat'}


        image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)


        if self.target_transform:
            label = self.target_transform(label)

        return image, torch.tensor(label).long()
    
    def __process_data(self):

        Airplane_images = os.listdir(os.path.join(self.data_dir, self.data_type, 'Airplane'))
        Automobile_images = os.listdir(os.path.join(self.data_dir, self.data_type, 'Automobile'))
        Bird_images = os.listdir(os.path.join(self.data_dir,self.data_type,'Bird'))
        Cat_images = os.listdir(os.path.join(self.data_dir,self.data_type,'Cat'))


        Airplane_images = [image_name for image_name in Airplane_images if '.png' in image_name]
        Automobile_images = [image_name for image_name in Automobile_images if '.png' in image_name]
        Bird_images = [image_name for image_name in Bird_images if ' .png' in image_name]
        Cat_images = [image_name for image_name in Cat_images if ' .png' in image_name]


        combined_images = Airplane_images + Automobile_images + Bird_images + Cat_images
        labels = [0]*len(Airplane_images) + [1]*len(Automobile_images) + [2]*len(Bird_images)

        return combined_images, labels
    
# if __name__ == '__main__':
#     data_dir = 'Images'
#     data_type = 'train'

#     transforms = t.Compose([t.Resize((224, 224)),
#                             t.ToTensor()])
    
#     dataset = MyDataset(data_dir, data_type, transform=transforms)

#     index = 4 
#     x, y = dataset[2] 
#     plt.imshow(x.permute(1,2,0))
#     plt.show()
#     print("input: ", x.shape)
#     print("label: ", y)



