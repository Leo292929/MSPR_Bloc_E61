#Import
from random import shuffle
import numpy as np
import os
from PIL import Image
from itertools import islice
from torch import tensor

###loading des datas ==> un tableau avec autant de tuple que d'image, avec dans chaque tuple ("le label",l'image en array numpy)
def loading_datas(chemin_dossier):
    
    data_brute = []
    classes = []

    for sous_dossier in os.listdir(chemin_dossier):

        chemin_sous_dossier = os.path.join(chemin_dossier, sous_dossier)

        classe = sous_dossier
        classes.append(classe)

        for fichier in os.listdir(chemin_sous_dossier):

            chemin_fichier = os.path.join(chemin_sous_dossier, fichier)
            image = Image.open(chemin_fichier)

            image_array = np.array(image, dtype=np.float32)

            image_array = image_array.reshape((image_array.shape[0])*(image_array.shape[1]),1)

            data_brute.append((one_hot_encoder(classes,classe),image_array))
    classes_encoded = []
    for e in classes:
        classes_encoded.append(one_hot_encoder(classes,e))

    return data_brute,classes_encoded


# pour chaque classe on save nsave element qui serviront pour la validation finale
def save_val_data(data,classes,nsave):
    val_data = []
    shuffle(data)
    for classe in classes:
        val_data = val_data + list(islice(filter(lambda x: x[0] == classe , data), nsave))
    for e in val_data:
        data.remove(e)
    return data,val_data
            


#separe les données en 3 groupe avec les proportion 70 15 15
def spliting(X,frac):
  trainr=frac
  
  Xtrain = X.sample(frac=trainr)
  Xtest = X.drop(Xtrain.index)

  return Xtrain,Xtest,
#ne sert pas

chemin_dossier = "Mammiferes_clean"

def one_hot_encoder(classes,classe):
    classe_encoded = [0.]*len(classes)
    i  = 0 
    for e in classes:
        if e == classe:
            classe_encoded[i]=1.
        i+=1
    return classe_encoded


#recupere les données loader et renvoie les données sour forme de tenseur

def to_tenseur(data,val_data,classes):
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
    data_brute,classes = loading_datas(chemin_dossier)

    data,val_data = save_val_data(data_brute,classes,3)
    print(1)
    list_label_tenseur,list_image_tenseur,val_data_tenseur,classes_tenseur = to_tenseur(data,val_data,classes)
    return list_label_tenseur,list_image_tenseur,val_data_tenseur,classes_tenseur



# x,y,z,t = load(chemin_dossier)

###on a donc une fonction qui prned un dossier( organiser avec un sous dossier par classe et des donne clean) et qui renvoie
###une liste avec les données pour la validation final et une liste data avec des tuple : ("le label",l'image en array numpy)
### le label est encoder 



