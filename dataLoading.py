#Import
from random import shuffle
import numpy as np
import os
from PIL import Image
from itertools import islice
from torch import tensor
import warnings
warnings.filterwarnings("ignore")
###loading des datas ==> un tableau avec autant de tuple que d'image, avec dans chaque tuple ("le label",l'image en array numpy)
def loading_datas(chemin_dossier):
    
    set_image= []
    set_label = []

    label = 1

    for sous_dossier in os.listdir(chemin_dossier):

        chemin_sous_dossier = os.path.join(chemin_dossier, sous_dossier)

        for fichier in os.listdir(chemin_sous_dossier):

            chemin_fichier = os.path.join(chemin_sous_dossier, fichier)
            image = Image.open(chemin_fichier)


            image_array = np.array(image, dtype=np.float32)

            try:
                image_array = np.transpose(image_array,(2, 0, 1))
                set_image.append(image_array.tolist())
                set_label.append(label)
            except:
                print(image_array.shape)
                print(fichier)     

        label+=1           


    return set_image,set_label


# pour chaque classe on save nsave element qui serviront pour la validation finale
def save_val_data(set_image,set_label,nb_save):
    val_data = []
    for i in range(max(set_label)-1):
        for j in range(nb_save):
            for e,f in zip(set_label,set_image):
                if e == i+1:
                    val_data.append([f,e])
                    set_label.remove(e)
                    set_image = [sub_list for sub_list in set_image if (sub_list != f)]
                    break
    return set_image,set_label,val_data



def merge_custom(set_image,set_label):
    set_train = []
    for e, f in zip(set_image,set_label):
        set_train.append((e,f))
    return set_train


#separe les données en 3 groupe avec les proportion 70 15 15
def spliting(X,frac):
  trainr=frac
  
  Xtrain = X.sample(frac=trainr)
  Xtest = X.drop(Xtrain.index)

  return Xtrain,Xtest,
#ne sert pas



def one_hot_encoder(classes,classe):
    classe_encoded = [0.]*len(classes)
    i  = 0 
    for e in classes:
        if e == classe:
            classe_encoded[i]=1.
        i+=1
    return classe_encoded


#recupere les données loader et renvoie les données sour forme de tenseur

def to_tenseur(data,val_data):
    list_label_tenseur,list_image_tenseur,val_data_tenseur,classes_tenseur = [],[],[],[]
    for e in classes:
        classes_tenseur.append(tensor(e))
    for e in val_data:
        val_data_tenseur.append((tensor(e[0]),tensor(e[1])))
    for e in data:
        list_label_tenseur.append(tensor(e[0]))
        list_image_tenseur.append(tensor(e[1]))
    return list_label_tenseur,list_image_tenseur,val_data_tenseur,classes_tenseur




def load(chemin_dossier):
    set_image,set_label = loading_datas(chemin_dossier)
    for image in set_image:
        for channel in image:
            for ligne in channel:
                for colonne in ligne:
                    colonne/=255.

    set_image,set_label,val_data = save_val_data(set_image,set_label,5)
    set_train = merge_custom(set_image,set_label)
    print(type(set_train))
    print(type(set_train[0]))
    print(type(set_train[0][0]))
    set_train[0] = [tensor(set_train[0][0]),set_train[0][1]]
    set_train[0]=tensor(set_train[0])
    val_data = tensor(val_data)
    print(set_train[0])
 #reste a fusionner set_label et set_image et a tout passer en tenseur

    # list_label_tenseur,list_image_tenseur,val_data_tenseur,classes_tenseur = to_tenseur(data,val_data,classes)
    # return list_label_tenseur,list_image_tenseur,val_data_tenseur,classes_tenseur

chemin_dossier = "Mammiferes_clean"

load(chemin_dossier)




###on a donc une fonction qui prned un dossier( organiser avec un sous dossier par classe et des donne clean) et qui renvoie
###une liste avec les données pour la validation final et une liste data avec des tuple : ("le label",l'image en array numpy)
### le label est encoder 


#encodage inutile
#3 channels