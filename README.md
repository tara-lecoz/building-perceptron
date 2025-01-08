# Contexte du projet

Le projet "Initiation au Deep Learning" vise à explorer les concepts fondamentaux de l'intelligence artificielle, en se concentrant sur le Machine Learning et plus particulièrement le Deep Learning. 

# Outils utilisés

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

# Veille

## ➔ Définissez les notions de Machine Learning et de Deep Learning puis comparez-les. A quel moment doit-on utiliser l’un plutôt que l’autre ?



## ➔ Faites une recherche sur les différentes applications du Deep Learning et présentez en 3. Vous pouvez vous inspirer de AI Experiments ou de OpenAI.

#### a) Reconnaissance d’images et de vidéos

Le Deep Learning permet aux ordinateurs de "voir" et de comprendre ce qu’il y a dans une image ou une vidéo. Par exemple, il peut reconnaître des objets, des visages ou des paysages.

Exemple : Google Photos utilise cette technologie pour classer automatiquement vos photos (par lieu, personne ou objet). OpenAI a aussi créé DALL-E, un outil qui génère des images à partir de descriptions textuelles.

#### b) Traitement du langage (comme les chatbots)

Le Deep Learning aide les machines à comprendre et à générer du texte. Par exemple, il peut traduire des langues, répondre à des questions ou écrire des articles.

Exemple concret : ChatGPT d’OpenAI est un chatbot qui peut discuter avec vous, écrire des textes ou même aider à coder. Google utilise aussi cette technologie pour améliorer son assistant vocal et ses recherches.

#### c) Voitures autonomes

Les voitures autonomes utilisent le Deep Learning pour "comprendre" leur environnement. Elles analysent les données des caméras et des capteurs pour éviter les obstacles et conduire en toute sécurité.

Exemple concret : Tesla utilise cette technologie pour ses voitures autonomes. OpenAI a aussi travaillé sur des simulations pour entraîner des voitures autonomes dans des environnements virtuels.

## 1. Qu’est ce qu’un Perceptron ? Quel est le lien entre un neurone biologique et un perceptron ?



## 2. Quelle est la fonction mathématique du Perceptron et son usage ? Définissez les termes de l’équation.



## 3. Donnez une ou plusieurs règles d’apprentissage du Perceptron.


   
## 4. Le perceptron utilise généralement une fonction d’activation, laquelle ?

Le perceptron est un des premiers modèles de réseaux de neurones. Il utilise une fonction d’activation simple qui fonctionne comme un interrupteur :

```
f(x) = { 
        1 si x ≥ 0 
        0 si x < 0 
      } 
```
En gros, si le résultat des calculs est positif ou nul, le perceptron "s’allume" (1). Sinon, il reste "éteint" (0).
   
## 5. Quel est le processus d'entraînement du Perceptron ?

L’entraînement du perceptron consiste à ajuster ses paramètres pour qu’il fasse moins d’erreurs. 

Voici les étapes :
* Initialisation : On commence par donner des valeurs aléatoires (ou à zéro) aux poids du perceptron.
* Calcul de la sortie : Pour chaque exemple d’entraînement, le perceptron fait une prédiction en fonction des données et des poids actuels.
* Comparaison : On compare la prédiction du perceptron avec la réponse correcte.
* Mise à jour des poids : Si la prédiction est fausse, on ajuste les poids pour améliorer la prédiction.
  * La formule est ``` wi =wi +η⋅(y−y^ )⋅xi ``` où :
    * wi  : le poids à ajuster,
    * η : la vitesse d’apprentissage (taux d’apprentissage),
    * y : la réponse correcte,
    * y^  : la prédiction du perceptron,
    * xi  : l’entrée.
* Répétition : On répète ces étapes jusqu’à ce que le perceptron fasse peu ou pas d’erreurs.
   
## 6. Quelles sont les limites du Perceptron ?

Le perceptron est un modèle simple, mais il a des limites.

* Problèmes linéaires seulement : Le perceptron ne peut résoudre que des problèmes où les données sont séparables par une ligne droite. Par exemple, il ne peut pas résoudre un problème comme le XOR (un cas où les données ne sont pas linéairement séparables).
* Sortie binaire : Le perceptron donne seulement une réponse "oui" (1) ou "non" (0). Il ne peut pas indiquer à quel point il est sûr de sa réponse.
* Dépendance au taux d’apprentissage : Si le taux d’apprentissage (η) est trop élevé, le perceptron peut ne pas converger. S’il est trop faible, l’apprentissage sera très lent. 
* Pas de couches cachées : Le perceptron n’a qu’une seule couche de neurones, ce qui limite sa capacité à résoudre des problèmes complexes. Les modèles modernes utilisent plusieurs couches pour mieux apprendre.
* Pas de convergence garantie : Si les données ne sont pas linéairement séparables, le perceptron peut ne jamais trouver une solution et continuer à faire des erreurs.   
   
## 7. Vous développez votre propre Perceptron à l’aide de Python en programmation orientée objet. Vous le testez sur des données factices générées de manière aléatoire.

cf fichier .py

# Les données et leur analyse

Dans le cadre de l'analyse des données, une approche rigoureuse a été adoptée pour explorer et comprendre les caractéristiques du jeu de données "Breast Cancer Wisconsin". 

Une analyse univariée a permis d'examiner individuellement chaque variable, en identifiant leurs distributions et leurs propriétés statistiques.

Ensuite, une analyse bivariée a été réalisée pour étudier les relations entre les variables, révélant d'éventuelles corrélations ou tendances.

Enfin, une analyse multivariée a été menée pour explorer les interactions complexes entre plusieurs variables simultanément, en s'appuyant sur des visualisations graphiques claires et informatives. 

Pour simplifier la structure des données et améliorer l'efficacité du modèle, une réduction de dimensionnalité a été effectuée à l'aide de l'Analyse en Composantes Principales (PCA). 

Cette méthode a permis de conserver l'essentiel de l'information tout en réduisant le nombre de variables, facilitant ainsi la modélisation et l'interprétation des résultats.

# Conclusion



# Bibliographie

