#OpenCV Version : opencv_python-4.5.4.58
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def ShiTomasi():
    img_kp = cv2.goodFeaturesToTrack(img_gray,500,0.01,10)
    img_kp = np.int0(img_kp)

    frag_kp = cv2.goodFeaturesToTrack(frag_gray,10,0.01,10)
    frag_kp = np.int0(frag_kp)

    for i in img_kp:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    plt.imshow(img),plt.show()

    for i in frag_kp:
        x,y = i.ravel()
        cv2.circle(frag,(x,y),3,255,-1)
    plt.imshow(frag),plt.show()

    # Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

    # On fait la correspondance entre le descripteur SIFT et le fragment
    matches = bf.match(img_kp, frag_kp)

def SIFT():
    sift = cv2.SIFT_create()

    train_keypoints, train_descriptor = sift.detectAndCompute(img_gray, None)
    test_keypoints, test_descriptor = sift.detectAndCompute(frag_gray, None)

    keypoints_without_size = np.copy(img)
    keypoints_with_size = np.copy(img)

    cv2.drawKeypoints(img, train_keypoints, keypoints_without_size, color = (0, 255, 0))
    cv2.drawKeypoints(img, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fx, plots = plt.subplots(1, 2, figsize=(20,10))

    plots[0].set_title("Points d'intérêts \"With Size\"")
    plots[0].imshow(keypoints_with_size, cmap='gray')

    plots[1].set_title("Train keypoints \"Without Size\"")
    plots[1].imshow(keypoints_without_size, cmap='gray')

    print("Nombre de points d'interets dans l'image: ", len(train_keypoints))
    print("Nombre de points d'interets dans le fragment: ", len(test_keypoints))

    # Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

    # On fait la correspondance entre le descripteur SIFT et le fragment
    matches = bf.match(train_descriptor, test_descriptor)

    # On veut les correspondances avec la plus petite distance
    matches = sorted(matches, key = lambda x : x.distance)
    print("\nNombre de points de correspondance avec le fragment : ", len(matches))

    best_matches = []
    for i in range(0,100):
        best_matches.append(matches[i])

    result = cv2.drawMatches(img, train_keypoints, frag_gray, test_keypoints, best_matches, frag_gray, flags = 2)

    # Les meilleurs correspondances
    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show()

#data - un ensemble d'observations
#modele - un modèle qui peut être ajusté à des données
#n - le nombre minimum de données nécessaires pour ajuster le modèle
#k - le nombre maximal d'itérations de l'algorithme
#t - une valeur seuil pour déterminer si une donnée correspond à un modèle
#d - le nombre de données proches des valeurs nécessaires pour faire valoir que le modèle correspond bien aux données
def ransac(data, model, n, k, t, d):

    iterator = 0
    meilleur_modele = None
    meilleur_ensemble_points = None
    meilleure_erreur = 999999999999999999999999999999999999999999999999999999999999999999999999 # Doit être l'infini

    while iterator < k:
        for i in range(0, n):
            points_aléatoires.append(random.choice(data))
        modèle_possible = None # ???
        ensemble_points = points_aléatoires

        for point in points_aléatoires:
            if point - modele_possible < t:
                ensemble_points.append(point)
        if len(ensemble_points) > d:
            modèle_possible = None #???
            erreur = None #???
            if erreur < meilleure_erreur:
                meilleur_modele = modèle_possible
                meilleur_ensemble_points = ensemble_points
                meilleure_erreur = erreur
        iterator++s

    # Return 
    # meilleur_modèle - les paramètres du modèle qui correspondent 
    # le mieux aux données (ou zéro si aucun bon modèle n'a été trouvé)
    # meilleur_ensemble_points - données à partir desquelles ce modèle a été estimé
    # meilleure_erreur - l'erreur de ce modèle par rapport aux données
    return [meilleur_modele, meilleur_ensemble_points, meilleure_erreur]

img = cv2.imread("Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg")
frag = cv2.imread("Michelangelo/frag_eroded/frag_eroded_0.png")

# Conversion en niveau de gris
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
frag_gray = cv2.cvtColor(frag, cv2.COLOR_RGB2GRAY)

# Affichage des images pour debug
#fx, plots = plt.subplots(1, 2, figsize=(20,10))
#plots[0].set_title("Training Image")
#plots[0].imshow(img)
#plots[1].set_title("Testing Image")
#plots[1].imshow(frag)

#Decommenter pour SIFT
#SIFT()

#Decommenter pour le Thomasi
ShiTomasi()

