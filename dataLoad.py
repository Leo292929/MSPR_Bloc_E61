import os
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import warnings
warnings.filterwarnings("ignore")

class CustomImageDataset(Dataset):

    def __init__(self, img_dir, transform=None, target_transform=None):

        listpath = []
        listlabel = []
        ### 
        label = 1
        for sous_dossier in os.listdir(img_dir):
            chemin_sous_dossier = os.path.join(img_dir, sous_dossier)
            for fichier in os.listdir(chemin_sous_dossier):
                chemin_fichier = os.path.join(chemin_sous_dossier, fichier)
                listpath.append(chemin_fichier)
                listlabel.append(label)
        label+=1

        dico = {'path':listpath,
                'label':listlabel}
        ###
        self.img_labels = pd.DataFrame(dico)

        self.img_dir = img_dir

        self.transform = transform
        self.target_transform = target_transform



    def __len__(self):

        return len(self.img_labels)



    def __getitem__(self, idx):

        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label





def load():

    chemin_dossier = "Mammiferes_clean"

    data = CustomImageDataset(chemin_dossier)


    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

