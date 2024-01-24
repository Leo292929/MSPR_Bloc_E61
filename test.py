import random

liste1 = [1, 2, 3, 4, 5]
liste2 = ['a', 'b', 'c', 'd', 'e']

# Utilisation d'une graine pour le générateur de nombres aléatoires
seed_value = 42
random.seed(seed_value)

# Mélange des listes
random.shuffle(liste1)
random.seed(seed_value)  # Réinitialisation de la graine pour assurer la reproductibilité
random.shuffle(liste2)

# Affichage des résultats
print("Liste 1 mélangée:", liste1)
print("Liste 2 mélangée:", liste2)