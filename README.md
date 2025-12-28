# Prévision Météorologique à Paris : Analyse, Modélisation et Étude Comparative

> **Matière :** Python pour la Data Science  
> **Auteurs :** Jesser ROKH & Khadija AMMAR  
> **Année :** 2025


## Problématique

> **Dans quelle mesure la confrontation entre l'inférence statistique (SARIMA) et l'apprentissage profond (LSTM) permet-elle d'appréhender la complexité du climat urbain, et quel est le degré de fiabilité de leurs prévisions face aux complexités et limites structurelles des séries temporelles ?**

Ce projet s'articule autour de trois phases principales :

1.  **Acquisition et traitement des données.**
2.  **Analyse exploratoire :** Étude statique et dynamique.
3.  **Modélisation et comparaison :** Mise en œuvre d'approches statistiques autorégressives face au Deep Learning.

---

## Architecture du Projet

Le projet est structuré en 4 étapes progressives :

| Ordre | Notebook | Objectif Principal |
| :--- | :--- | :--- |
| **01** | [Data Collection](./01_data_collection.ipynb) | Acquisition via API Open-Météo et définition du périmètre temporel. |
| **02** | [EDA (Exploratory Data Analysis)](./02_eda.ipynb) | Nettoyage, interpolation, analyse des distributions et corrélations. |
| **03** | [Time Series Analysis](./03_time_series_analysis.ipynb) | Analyse spectrale, saisonnalité, stationnarité et autocorrélation. |
| **04** | [Modelling](./04_modelling.ipynb) | Feature Engineering, entraînement (SARIMA, LSTM, CNN-LSTM) et évaluation. |

## Rapport Final

Vous pouvez accéder au notebook de synthèse finale en cliquant sur le lien ci-dessous :

**[Rapport Final & Synthèse des Résultats](./rendu_final.ipynb)**


**Note de Lecture Importante** : **Le rapport final est un travail de synthèse.** Il a pour vocation de présenter uniquement les résultats finaux les plus pertinents et la confrontation des modèles.
Pour une compréhension approfondie, **tous les détails méthodologiques, les justifications mathématiques, les analyses intermédiaires et le code complet sont expliqués de manière exhaustive dans les notebooks correspondants ci-dessus.**
