# Prédiction du trafic routier en Île‑de‑France

## Contexte et motivation

Ce projet personnel est né d’une observation simple : les applications de navigation comme Waze ou Google Maps peuvent être contraintes par des réglementations écologiques ou des critères commerciaux, et ne proposent pas toujours l’itinéraire réellement le plus rapide. En partant de ce constat, j’ai décidé de développer un outil autonome capable d’estimer les temps de parcours en voiture en Île‑de‑France, en combinant données réelles et apprentissage automatique.

## Objectifs principaux

* Estimer le temps de trajet entre deux adresses avec une précision accrue, en tenant compte des variations horaires, du week‑end et des caractéristiques du réseau.
* Offrir une cartographie interactive de l’itinéraire, exportable en GPX ou HTML.

## Architecture du script

Le script `TrajIDF.py` s’organise en plusieurs étapes :

1. **Acquisition des données**

   * Extraction du réseau routier via OSMnx.
   * Collecte des comptages horaires depuis l’API Paris Open Data.
   * Intégration des flux en temps réel de Sytadin (données XML).
   * Génération de données synthétiques pour pallier les capteurs manquants.

2. **Prétraitement et feature engineering**

   * Construction et mise en cache du graphe routier de l’Île‑de‑France (format pickle).
   * Calcul empirique des vitesses à partir du débit et de l’occupation.
   * Création de variables temporelles (heures cycliques, indicateurs de pointe/week‑end) et spatio‑structurelles (longueur, nombre de voies, type de route, sens unique, ponts, tunnels).
   * Conception d’interactions pertinentes (par exemple heure × type de route).

3. **Modélisation**

   * Division des données en jeux d’entraînement et de test, stratifiée par heure et jour.
   * Normalisation des variables.
   * Trois modèles : XGBoost, Gradient Boosting, Random Forest, pondérés selon leur performance.
   * Calibration isotonique pour corriger les biais de prédiction.
   * Évaluation via MAE, RMSE et R² avant/après calibration.

4. **Calcul d’itinéraire**

   * Géocodage des adresses avec Nominatim.
   * Construction d’un graphe temporel pondéré par les durées estimées.
   * Exécution de l’algorithme de Dijkstra pour déterminer le chemin le plus rapide.
   * Restitution : temps total, distance, vitesse moyenne et feuille de route.

5. **Visualisation**

   * Carte interactive Folium avec polylignes et marqueurs.
   * Panneau récapitulatif et export GPX/HTML.

## Choix méthodologiques

Après avoir normalisé les variables et effectué une validation croisée à 5 folds, trois algorithmes de boosting et de forêt aléatoire ont été retenus :

* **XGBoost** 
* **Gradient Boosting (scikit‑learn)**
* **Random Forest**

Les prédictions de chaque composant sont combinées par pondération croisée, déterminée à partir de la performance sur le jeu de validation (50 % XGBoost, 30 % GBM, 20 % RF). Une calibration isotonique a ensuite été appliquée pour réajuster la relation prédite versus observée, réduisant ainsi les biais systématiques identifiés lors des heures de pointe.

## Difficultés et solutions

* **Graphe volumineux** : plus de 500 000 arêtes pour l’Île‑de‑France. Solution : cache local et chargement progressif.
* **Données manquantes ou déséquilibrées** : capteurs mal répartis. Solution : génération synthétique avec tolérance spatiale.
* **Performance de calcul** : prédiction lente sur tout le réseau. Solution : vectorisation des calculs et option multithreading.
* **Gestion mémoire** : pics lors du chargement complet. Solution : extraction segmentée et mode dégradé (Paris seul).

## Performances de l’estimateur

Lors des tests finaux sur le jeu de validation, le modèle combiné a atteint les scores suivants :

* **Erreur absolue moyenne (MAE)** : 1,7 minute
* **Erreur quadratique moyenne (RMSE)** : 2,4 minutes
* **Coefficient de détermination (R²)** : 0,88

Après calibration isotonique, le RMSE est passé de 2,4 à 2,1 minutes et le MAE de 1,7 à 1,5 minute, attestant de l’efficacité de l’ajustement post‑prédiction.

## Installation et exécution

Pour installer les dépendances :

```bash
py -m pip install -r requirements.txt
```

Lancer le script :

```bash
py TrajIDF.py
```

Deux modes sont proposés : entraînement complet ou chargement d’un modèle existant. Si vous souhaitez ré-entrainer le modèle ou re-télécharger le graphe routier, il est préférable de supprimer les fichiers modele_trafic_idf.pkl et graphe_routier_idf.pkl. Pour une première utilisation, choisissez entraînement complet.

## Structure du dépôt

Le dépôt contient :

```
TrajIDF.py          # Script principal
requirements.txt    # Dépendances
README.md           # Ce document
```

## Références

* Documentation OSMnx
* Guide XGBoost (régularisation et ensembles)
* Méthodes d’ensemble dans scikit‑learn

**Auteur** : Mathis R.
**Licence** : Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
