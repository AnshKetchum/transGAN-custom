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
    def __init__(self, YOLO_DIR, transform, IMAGE_SIZE = 256):

        #Initialize parameters from constructor
        self.YOLO_DIR = YOLO_DIR
        self.df = pd.read_csv(os.path.join(self.YOLO_DIR, 'raw_dataset' ,'images-info.csv'))
        self.transform = transform

    #Returns length of the dataset
    def __len__(self):
        return len(self.df)

    #Gets a specific item from the dataset
    def __getitem__(self, index):

        #Get the ith row of the CSV - has all the information
        row = self.df.iloc[index]

        #Read the image
        X = PIL.Image.open(os.path.join(self.YOLO_DIR, 'raw_dataset', row["dataset-path"])).convert("RGB")
        X = self.transform(X)

        #Read the labels
        y = row["label"]

        return X,y

print("Dataset class compiled.")