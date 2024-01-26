from load import load
from modeleDef import function_modele
from torch import nn,tensor
import random
from fun import *
from datacleanning import creer_copie_clean


chemin_dossier_entree = "Mammifères"
chemin_dossier_sortie = "Mammiferes_clean"

creer_copie_clean(chemin_dossier_entree,chemin_dossier_sortie)



list_label_t,list_image_t,val_data_t,classes_t = load("Mammiferes_clean")





nb_epoch = 1

lr = 0.01

gamma = 1

nb_classe = len(classes_t)

model,optimizer = function_modele()

#liste sauvegardant les loss et l'accuracy pour pouvoir analyser l'evolution
liste_evo_loss = []
liste_evo_acc = []





for i in range(nb_epoch):

    train_list_label_t , test_list_label_t , train_list_image_t , test_list_image_t = train_test(list_label_t,list_image_t)

    for j in range(len(train_list_label_t)):

        print(train_list_image_t[j])
        output = model(train_list_image_t[j])

        loss = nn.loss=nn.functional.cross_entropy(output,train_list_label_t[j])
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


    #uniquement pour le controle
    list_prediction_test_t = model(test_list_image_t)
    oh_list_predi_test_t = []
    for e in list_prediction_test_t:
        oh_list_predi_test_t.append(one_hot_max(e))
        
    accuracy = calc_accuracy(oh_list_predi_test_t,test_list_label_t)

    liste_evo_acc.append(accuracy)
    liste_evo_loss.append(loss)

    print("epoch {}\t loss : {}\t accuracy : {}\t ".format(i,loss,accuracy))



    
