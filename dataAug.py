from PIL import Image
import numpy as np
import os
import random as rd
from PIL import ImageFilter,ImageOps

def augment_data(chemin_dossier_entree,chemin_dossier_sortie,nbFichierFinal = 40):
    for sous_dossier in os.listdir(chemin_dossier_entree):
        num = 0
        chemin_sous_dossier = os.path.join(chemin_dossier_entree, sous_dossier)
        chemin_sous_dossier_sortie = os.path.join(chemin_dossier_sortie, sous_dossier)
        
        if not os.path.exists(chemin_sous_dossier_sortie):
            os.makedirs(chemin_sous_dossier_sortie)

        turn = 1
        while len(os.listdir(chemin_sous_dossier_sortie))<nbFichierFinal:
            for fichier in os.listdir(chemin_sous_dossier):
                if len(os.listdir(chemin_sous_dossier_sortie))>nbFichierFinal:
                    break

                chemin_fichier = os.path.join(chemin_sous_dossier, fichier)

                chemin_fichier_sortie = os.path.join(chemin_sous_dossier_sortie,sous_dossier+str(num+1)+".jpg")
                
                image = Image.open(chemin_fichier)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')

                if turn == 1:
                    pass
                else:
                    #randomCrop
                    largeur, hauteur = image.size
                    rx1 = rd.randint(0,200)
                    rx2 = rd.randint(0,200)
                    ry1 = rd.randint(0,200)
                    ry2 = rd.randint(0,200)


                    x1 = 0 + rx1
                    y1 = 0 + rx2
                    x2 = largeur - ry1
                    y2 = hauteur - ry2

                    image = image.crop((x1, y1, x2, y2))
                    #randomRotation
                    image = image.rotate(rd.randint(0,360))
                    #randomMiroir
                    p = rd.random()
                    if p>0.66:
                        pass
                    elif p<0.33:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    #randomfilter
                    p = rd.random()
                    if p>0.5:
                        pass
                    elif p<0.25:
                        image = image.filter(ImageFilter.BLUR)
                    else:
                        image = image.filter(ImageFilter.DETAIL)
                    #rondomNegatif
                    if rd.random()<0.5:
                        image = ImageOps.invert(image)


                image.save(chemin_fichier_sortie, 'jpeg')


                print(chemin_fichier_sortie)
                num+=1

            turn+=1


augment_data("Mammiferes","Mammiferes_augmente")