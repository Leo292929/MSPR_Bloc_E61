from load import load
#from modeleDef import function_modele
from torch import nn,tensor
import random
from fun import *
from dataCleanning import creer_copie_clean
from dataAug import augment_data


augData = False
cleanData = False

if augData:

    chemin_dossier_entree = "Mammiferes"
    chemin_dossier_sortie = "Mammiferes_augmente"

    augment_data(chemin_dossier_entree,chemin_dossier_sortie)

    print("data augmentée")

if cleanData:
    chemin_dossier_entree = "Mammiferes_augmente"
    chemin_dossier_sortie = "Mammiferes_clean"

    creer_copie_clean(chemin_dossier_entree,chemin_dossier_sortie)


    print("data cleannée")







list_label_t,list_image_t,val_data_t,classes_t = load("Mammiferes_clean")





nb_epoch = 1
lr = 0.01
gamma = 1
nb_classe = len(classes_t)


#liste sauvegardant les loss et l'accuracy pour pouvoir analyser l'evolution
liste_evo_loss = []
liste_evo_acc = []







    
