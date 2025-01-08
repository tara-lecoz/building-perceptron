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

``` f(x)={10 si x≥0si x<0 ```

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
