'''
This cell contains the Dataset object 
for the Pytorch Dataloader API 
'''

import os
import torch
import PIL
import pandas as pd
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, IMAGE_SIZE = 256):

        #Initialize parameters from constructor
        self.df = pd.read_csv(os.path.join(YOLO_DIR, 'raw_dataset' ,'images-info.csv'))
        self.transforms = transforms.Compose( 
            [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), #Convert to tensor
            transforms.Normalize( mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])] #Normalize - subtract by the mean and divide by the standard deviation
        )


    #Returns length of the dataset
    def __len__(self):
        return len(self.df)

    #Gets a specific item from the dataset
    def __getitem__(self, index):

        #Get the ith row of the CSV - has all the information
        row = self.df.iloc[index]

        #Read the image
        X = PIL.Image.open(os.path.join(YOLO_DIR, 'raw_dataset', row["dataset-path"])).convert("RGB")
        X = self.transforms(X)

        #Read the labels
        y = row["label"]

        return X,y

print("Dataset class compiled.")