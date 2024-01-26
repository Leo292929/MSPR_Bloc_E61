from PIL import Image
from datacleanning import normalize_grayscale_image
import os
from modeleDef import ResNet20
import torch
from interactionSQL import result2Info,insertIn2Db





def clean(chemin):

    cheminBureau = os.path.join(os.path.expanduser('~'), 'Desktop')
    cheminClean = os.path.join(cheminBureau, 'imageClean.jpeg')

    image = Image.open(chemin)
            
    #dim du crop
    width, height = image.size
    if width > height:
        left = (width - height) // 2
        top = 0
        right = left + height
        bottom = height
    else:
        left = 0
        top = (height - width) // 2
        right = width
        bottom = top + width


    image = image.crop((left, top, right, bottom))
    image = image.resize((256,256))
    image = image.convert('L')

    image = normalize_grayscale_image(image)

    image.save(cheminClean, 'JPEG')

    return cheminClean


def chemin2tenseur(chemin):
    return tenseur





idUser = input("quel est votre id Utilisateur")
chemin = input("veuillez saisir le chemin de l'image à analyser")
cheminClean = clean(chemin)
tenseur = chemin2tenseur(cheminClean)

model = ResNet20()
model.load_state_dict(torch.load("modeleState.pt"))

resultat = model.result(tenseur)



result2Info(resultat)

insertIn2Db(chemin,resultat,idUser)


