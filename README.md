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

L’apprentissage automatique (machine learning en anglais) est un champ d’étude de l’intelligence artificielle qui vise à donner aux machines la capacité d’ apprendre à partir de données, via des modèles mathématiques. Plus précisément, il s’agit du procédé par lequel les informations pertinentes sont tirées d’un ensemble de données d’entraînement.
Le but de cette phase est l’obtention des paramètres d’un modèle qui atteindront les meilleures performances, notamment lors de la réalisation de la tâche attribuée au modèle. 

L'apprentissage automatique comporte généralement deux phases. La première consiste à estimer un modèle à partir de données, appelées observations, qui sont disponibles et en nombre fini, lors de la phase de conception du système. L'estimation du modèle consiste à résoudre une tâche pratique, telle que traduire un discours, estimer une densité de probabilité, reconnaître la présence d'un chat dans une photographie ou participer à la conduite d'un véhicule autonome. Cette phase dite « d'apprentissage » ou « d'entraînement » est généralement préalable à l'utilisation pratique du modèle. La seconde phase est la mise en production : le modèle étant déterminé, de nouvelles données peuvent alors être soumises afin d'obtenir le résultat correspondant à la tâche souhaitée.

Le deep learning est une discipline de l’intelligence artificielle capable d’analyser des données non structurées comme des images, des videos, du texte… C’est un système qui repose sur plusieurs couches de réseaux de neurones reliées entre elles, un peu comme le cerveau humain. Les réseaux sont composés de dizaines voir de centaines de “couches” de neurones, chacune recevant et interprétant les informations de la couche précédente. Plus il y a de couches, plus l’apprentissage est profond. 

Computer vision : Elle donne à une machine la capacité de voir et ainsi permettre l’analyse et l’interprétation d’images ou vidéos. Elle est utilisée dans différents secteurs comme par exemple la santé en diagnostiquant automatiquement une radiographie.

Natural Language Processing : Il donne quant à lui la possibilité à une machine de comprendre et d’interpréter le langage humain. Ainsi, cette technique de l’intelligence artificielle est utilisée dans le domaine de la traduction automatique.

Le Machine Learning et le Deep Learning sont deux types d’intelligence artificielle. Le Machine Learning est une IA capable de s’adapter automatiquement avec une interférence humaine minimale, et le Deep Learning est un sous-ensemble du Machine Learning utilisant les réseaux de neurones pour mimer le processus d’apprentissage du cerveau humain. Plusieurs différences majeures séparent ces deux concepts. Le Deep Learning requiert de plus larges volumes de données d’entraînement, mais apprend de son propre environnement et de ses erreurs.Au contraire, le Machine Learning permet l’entraînement sur des jeux de données moins vastes, mais requiert davantage d’intervention humaine pour apprendre et corriger ses erreurs.

Dans le cas du Machine Learning, un humain doit intervenir pour labelliser les données et indiquer leurs caractéristiques. Un système Deep Learning tente au contraire d’apprendre ces caractéristiques sans intervention humaine. Par exemple, pour la reconnaissance faciale, le programme de Deep Learning apprend d’abord à détecter et reconnaître les bordures et les lignes du visage. Il apprend ensuite les parties les plus importantes des visages, et finalement la représentation générale des visages. Ceci requiert d’immenses volumes de données, mais la probabilité de réussite augmente au fil de l’entraînement. L’approche est radicalement différente. Les algorithmes de Machine Learning tendent à séparer les données en plusieurs parties, qui sont ensuite combinées pour proposer un résultat ou une solution. De leur côté, les systèmes Deep Learning considèrent un problème dans son entièreté.
Le Machine Learning nécessite un temps d’entraînement plus court, mais son niveau de précision est plus faible. Le Deep Learning permet à la machine de réaliser des corrélations complexes et non linéaires entre les données.

L’entraînement Deep Learning est beaucoup plus long à cause de l’importante quantité de données à traiter, et des nombreux paramètres et formules mathématiques impliqués. Un système Machine Learning peut être entraîné en quelques secondes ou quelques heures, tandis que le Deep Learning peut nécessiter des semaines.
Enfin, le Machine Learning permet l’entraînement sur un CPU (unité de traitement centrale) tandis que le Deep Learning requiert un GPU (unité de traitement graphique). Ce puissant hardware est indispensable pour traiter les larges volumes de données et effectuer les calculs complexes des algorithmes.Compte tenu de leurs différences, le Machine Learning et le Deep Learning sont utilisés pour différentes applications. Le Machine Learning est exploité par les programmes prédictifs de la finance ou de la météo, les identificateurs de spam dans les emails, ou encore les programmes visant à concevoir des traitements personnalisés pour les malades.
Le Deep Learning est utilisé pour les recommandations des services de streaming, la reconnaissance faciale, mais également pour les véhicules autonomes. Grâce aux réseaux de neurones, les voitures sont capables de déterminer les objets à éviter, de reconnaître les feux tricolores et les panneaux, et de savoir quand accélérer ou ralentir.


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

Un Perceptron est un neurone artificiel, et donc une unité de réseau de neurones. Il effectue des calculs pour détecter des caractéristiques ou des tendances dans les données d’entrée.
Il s’agit d’un algorithme pour l’apprentissage supervisé de classificateurs binaires. C’est cet algorithme qui permet aux neurones artificiels d’apprendre et de traiter les éléments d’un ensemble de données.
Le Perceptron joue un rôle essentiel dans les projets de Machine Learning. Il est massivement utilisé pour classifier les données, ou en guise d’algorithme permettant de simplifier ou de superviser les capacités d’apprentissage de classificateurs binaires.
Il s'agit d'une simplification d'un neurone biologique, conçue pour imiter la manière dont le cerveau traite l'information


## 2. Quelle est la fonction mathématique du Perceptron et son usage ? Définissez les termes de l’équation.

Perceptron : Introduction et Fonction Mathématique

Un Perceptron est un neurone artificiel, et donc une unité de réseau de neurones. Il effectue des calculs pour détecter des caractéristiques ou des tendances dans les données d’entrée.

Il s’agit d’un algorithme pour l’apprentissage supervisé de classificateurs binaires. Cet algorithme permet aux neurones artificiels d’apprendre et de traiter les éléments d’un ensemble de données.

Le Perceptron joue un rôle essentiel dans les projets de Machine Learning. Il est massivement utilisé pour classifier les données ou pour superviser et simplifier les capacités d’apprentissage de classificateurs binaires. Il s'agit d'une simplification d'un neurone biologique, conçue pour imiter la manière dont le cerveau traite l'information.

Quelle est la fonction mathématique du Perceptron et son usage ?

Le perceptron est un algorithme de classification binaire utilisé dans le domaine de l'apprentissage automatique. Il s'agit d'un modèle de neurone artificiel qui :

1. Prend un vecteur d'entrée.
2. Effectue une somme pondérée de ces entrées.
3. Applique une fonction d'activation.
4. Produit une sortie binaire.

Fonction mathématique du Perceptron

La fonction mathématique du perceptron peut être décrite par l'équation suivante :

$$
y = f(\mathbf{w} \cdot \mathbf{x} + b)
$$

Termes de l'équation

- **\( \mathbf{x} \)** : Vecteur d'entrée. Il représente les caractéristiques ou les attributs de l'exemple que l'on souhaite classer. Par exemple, si vous avez trois caractéristiques, \( \mathbf{x} = [x_1, x_2, x_3] \).
- **\( \mathbf{w} \)** : Vecteur de poids. Chaque poids \( w_i \) est associé à une caractéristique \( x_i \) et détermine l'importance de cette caractéristique dans la décision finale. \( \mathbf{w} = [w_1, w_2, w_3] \).
- **\( b \)** : Biais (ou seuil). C'est un terme constant qui permet de décaler la fonction d'activation. Il aide le perceptron à s'ajuster pour des données qui ne sont pas centrées à l'origine.
- **\( \mathbf{w} \cdot \mathbf{x} \)** : Produit scalaire entre le vecteur de poids et le vecteur d'entrée. Cela donne une somme pondérée des entrées.
- **\( f \)** : Fonction d'activation. Dans le perceptron classique, il s'agit généralement de la fonction de seuil (ou fonction signe) qui produit une sortie binaire. Par exemple :

$$
f(z) = 
\begin{cases} 
1 & \text{si } z \geq 0 \\
0 & \text{si } z < 0 
\end{cases}
$$

où \( z = \mathbf{w} \cdot \mathbf{x} + b \).

- **\( y \)** : Sortie du perceptron. C'est la prédiction binaire du modèle, soit 0 soit 1, indiquant à quelle classe l'exemple d'entrée appartient.

Le perceptron est utilisé pour des tâches de classification binaire, où l'objectif est de séparer les données en deux classes distinctes. Il est particulièrement efficace pour les problèmes linéairement séparables, c'est-à-dire lorsque les deux classes peuvent être séparées par une ligne droite (ou un hyperplan dans des dimensions supérieures).
Bien que le perceptron soit un modèle simple, il a jeté les bases pour des modèles plus complexes comme les réseaux de neurones multicouches (MLP) et les réseaux de neurones profonds. Dans les cas où les données ne sont pas linéairement séparables, des techniques comme le perceptron multicouche ou l'utilisation de noyaux (dans le cadre des machines à vecteurs de support) peuvent être employées pour améliorer la capacité de classification.

3. Donnez une ou plusieurs règles d’apprentissage du Perceptron.

L'apprentissage du perceptron repose sur un algorithme itératif qui ajuste les poids et le biais pour minimiser les erreurs de classification sur un ensemble d'entraînement. Voici les étapes clés de la règle d'apprentissage du perceptron :

Règle d'apprentissage du Perceptron

1. **Initialisation** :
   - Commencez par initialiser les poids \(\mathbf{w}\) et le biais \(b\) à des petites valeurs aléatoires ou à zéro.

2. **Pour chaque exemple d'entraînement \((\mathbf{x}^{(i)}, y^{(i)})\)** :
   - **Calcul de la sortie** :
     - Calculez la sortie prédite \(\hat{y}^{(i)}\) en utilisant l'équation :
       \[
       \hat{y}^{(i)} = f(\mathbf{w} \cdot \mathbf{x}^{(i)} + b)
       \]
     - Où \(f\) est la fonction de seuil.

   - **Mise à jour des poids et du biais** :
     - Si l'exemple est mal classé (\(\hat{y}^{(i)} \neq y^{(i)}\)), mettez à jour les poids et le biais selon les règles suivantes :
       \[
       \mathbf{w} \leftarrow \mathbf{w} + \eta (y^{(i)} - \hat{y}^{(i)}) \mathbf{x}^{(i)}
       \]
       \[
       b \leftarrow b + \eta (y^{(i)} - \hat{y}^{(i)})
       \]
     - Où \(\eta\) est le taux d'apprentissage, un hyperparamètre qui détermine la taille des ajustements effectués à chaque étape.

3. **Répétition** :
   - Répétez le processus pour plusieurs itérations (ou époques) sur l'ensemble de données d'entraînement jusqu'à ce que les poids convergent (c'est-à-dire que les erreurs de classification soient minimisées) ou qu'un nombre maximal d'itérations soit atteint.

Remarques :

- **Taux d'apprentissage (\(\eta\))** : Un taux d'apprentissage trop élevé peut entraîner une convergence instable, tandis qu'un taux trop faible peut ralentir le processus d'apprentissage. Il est souvent nécessaire de tester plusieurs valeurs pour trouver le taux optimal.

- **Convergence** : Le perceptron converge uniquement si les données sont linéairement séparables. Si ce n'est pas le cas, l'algorithme peut ne jamais trouver une solution parfaite.

- **Extension aux problèmes non linéaires** : Pour traiter des problèmes non linéairement séparables, des extensions comme le perceptron multicouche (réseaux de neurones) ou l'utilisation de noyaux (dans le cadre des machines à vecteurs de support) peuvent être nécessaires.

L'algorithme du perceptron est simple mais puissant pour des problèmes de classification linéaire, et il a inspiré de nombreux développements dans le domaine de l'apprentissage automatique.

   
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

