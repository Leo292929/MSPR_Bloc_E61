# import random

# liste1 = [1, 2, 3, 4, 5]
# liste2 = ['a', 'b', 'c', 'd', 'e']

# # Utilisation d'une graine pour le générateur de nombres aléatoires
# seed_value = 42
# random.seed(seed_value)

# # Mélange des listes
# random.shuffle(liste1)
# random.seed(seed_value)  # Réinitialisation de la graine pour assurer la reproductibilité
# random.shuffle(liste2)

# # Affichage des résultats
# print("Liste 1 mélangée:", liste1)
# print("Liste 2 mélangée:", liste2)
from geopy.geocoders import Nominatim

def get_location_from_address(address):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(address)
    return location

address = "26 Rue Russeil,France"
location = get_location_from_address(address)
print("Coordonnées géographiques de Mountain View, CA :", (location.latitude, location.longitude))