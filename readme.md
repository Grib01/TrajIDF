# Prédiction du trafic routier en Île‑de‑France


Ce projet personnel est né d’une observation simple : les applications de navigation comme Waze ou Google Maps peuvent être contraintes par des réglementations écologiques ou des critères commerciaux, et ne proposent pas toujours l’itinéraire réellement le plus rapide. En partant de ce constat, j’ai décidé de développer un outil autonome capable d’estimer les temps de parcours en voiture en Île‑de‑France, en combinant données réelles et apprentissage automatique.

## Objectifs principaux

* Estimer le temps de trajet entre deux adresses avec une précision accrue, en tenant compte des variations horaires, du week‑end et des caractéristiques du réseau.
* Offrir une cartographie interactive de l’itinéraire, exportable en GPX ou HTML.

## Architecture du script

Le script `TrajIDF.py` s’organise comme suit :

1. **Acquisition des données**

   * Extraction du réseau routier via OSMnx.
   * Collecte des comptages horaires depuis l’API Paris Open Data.
   * Intégration des flux en temps réel de Sytadin (données XML).
   * Génération de données synthétiques pour pallier les capteurs manquants.
   * Construction et mise en cache du graphe routier de l’Île‑de‑France (format pickle).
   * Calcul empirique des vitesses à partir du débit et de l’occupation.
   * Création de variables temporelles (heures cycliques, indicateurs de pointe/week‑end) et spatio‑structurelles (longueur, nombre de voies, type de route, sens unique, ponts, tunnels).
   * Conception d’interactions pertinentes (par exemple heure × type de route).
   * Division des données en jeux d’entraînement et de test, stratifiée par heure et jour.
   * Normalisation des variables.
   * Trois modèles : XGBoost, Gradient Boosting, Random Forest.
   * Calibration isotonique pour corriger les biais de prédiction.
   * Évaluation via MAE, RMSE et R² avant/après calibration.
   * Géocodage des adresses avec Nominatim.
   * Construction d’un graphe temporel pondéré par les durées estimées.
   * Exécution de l’algorithme de Dijkstra pour déterminer le chemin le plus rapide.
   * Restitution : temps total, distance, vitesse moyenne et feuille de route.
   * Carte interactive Folium avec polylignes et marqueurs.
   * Panneau récapitulatif et export GPX/HTML.

## Choix méthodologiques

Après avoir normalisé les variables et effectué une validation croisée à 5 folds, trois algorithmes de boosting et de forêt aléatoire ont été retenus :

* **XGBoost** 
* **Gradient Boosting**
* **Random Forest**

Les prédictions de chaque composant sont combinées par pondération croisée, déterminée à partir de la performance sur le jeu de validation (50 % XGBoost, 30 % GBM, 20 % RF). Une calibration isotonique a ensuite été appliquée pour réajuster la relation prédite versus observée, réduisant ainsi les biais systématiques identifiés lors des heures de pointe.

Lors des tests finaux sur le jeu de validation, le modèle combiné a atteint les scores suivants :

* **Erreur absolue moyenne (MAE)** : 1,7 minute
* **Erreur quadratique moyenne (RMSE)** : 2,4 minutes
* **Coefficient de détermination (R²)** : 0,88

Après calibration isotonique, le RMSE est passé de 2,4 à 2,1 minutes et le MAE de 1,7 à 1,5 minute.


**Auteur** : Mathis R.
**Licence** : Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
