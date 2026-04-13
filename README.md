# 🏥 Modèle de Prédiction du Risque de Diabète

## 🔍 Modèle de Régression Logistique

Ce projet implémente un pipeline complet de machine learning permettant de prédire le risque de diabète (Faible vs Haut risque) à partir de plus de 30 variables cliniques, démographiques et comportementales.
Il suit une architecture modulaire, scalable, documentée, et prête pour GitHub.

## 🎯 Objectif du projet

- **Algorithme** : Régression Logistique (Classification Binaire)
- **Cible** : Prédire Haut Risque vs Faible Risque de diabète
- **Précision** : 97%
- **Dataset** : Kaggle – Diabetes Risk Prediction Dataset

## 📁 Structure du Projet

```
prediction-risque-diabetes/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances du projet
├── .gitignore                         # Règles git ignore
├── data/
│   └── diabetes_risk_dataset.cvs      # L'esnsemble des données (data set)
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration des constants
│   ├── data_loader.py                 # Charge les données du fichier CSV
│   ├── preprocessing.py               # Néttoyage et encodage des données
│   ├── eda.py                         # Analyse exploratoire des données
│   ├── model.py                       # Le model 
├── notebooks/
│   └── diabetes_analysis.ipynb       
└── results/
    ├── confusion_matrix.png
    ├── correlation_heatmap.png
    └── metrics.txt

```

## 🚀 Démarrage rapide

### 1. Cloner le projet
```bash
git clone https://github.com/yourusername/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer le pipeline complet
```bash
python main.py
```

## 📚 Décomposition du pipeline

### Étape 1 : Chargement des données (data_loader.py)
 - Charge le fichier CSV dans un DataFrame pandas
 - Valide la structure du dataset

### Étape 2 : Analyse exploratoire (eda.py)
 - Plus de 20 visualisations
 - Analyse des corrélations
 - Résumés statistiques

### Étape 3 : Prétraitement des données (preprocessing.py)
 - Gestion des valeurs manquantes / doublons
 - Encodage One-Hot des variables catégorielles
 - Normalisation des features avec StandardScaler
 - Séparation train/test (80/20)

### Étape 4 : Entraînement du modèle (model.py)
 - Entraîne la Régression Logistique
 - Génère les prédictions
 - Calcule l’accuracy et les métriques

### Étape 5 : Évaluation
 - Matrice de confusion
 - Rapport de classification (Précision, Rappel, F1-score)
 - Visualisation des performances

## 📊 Key Features

**Clinical Variables**:
- Age, BMI, Waist Circumference
- Fasting Glucose Level, HbA1c Level
- Blood Pressure, Insulin Level
- Triglycerides, Cholesterol

**Lifestyle Variables**:
- Physical Activity Level
- Daily Calorie Intake
- Sugar Intake
- Sleep Hours
- Stress Level

**Demographic**:
- Gender
- Family History of Diabetes

## 📈 Expected Results

```
Précision du modèle : 97%

Rapport de classification :
              precision    recall  f1-score   support
         0       0.97      0.97      0.97      1000
         1       0.97      0.97      0.97      1000

macro avg       0.97      0.97      0.97      2000
weighted avg    0.97      0.97      0.97      2000

```

## 🔧 Configuration

Edit `src/config.py` to modify:
- Test set size (default: 0.2 = 20%)
- Random state (for reproducibility)
- Model hyperparameters
- Plot styling

```python
# src/config.py
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 2000
FIGURE_SIZE = (10, 6)

```

## 📦 Dépendances
 - pandas >= 1.3.0
 - numpy >= 1.21.0
 - matplotlib >= 3.4.0
 - seaborn >= 0.11.0
 - scikit-learn >= 1.0.0

Voir `requirements.txt` pour la liste complète.

## 💡 Model Insights

### 💡 Insights du modèle
#### Principales conclusions :
1. **HbA1c** — Meilleur prédicteur du risque de diabète
2. **Glycémie à jeun** — Forte corrélation avec la catégorie de risque
3. **IMC & Tour de taille** — Indicateurs majeurs d’obésité
4. **Activité physique** — Facteur protecteur
5. **Stress & Sommeil** — Influence significative

### Corrélations importantes :
 - IMC ↔ Tour de taille : 0.92
 - Glycémie ↔ HbA1c : 0.85
 - Sucre consommé ↔ Glycémie : 0.78

## ⚠️ Points importants
1. **Aucune fuite de données** : le scaler est ajusté uniquement sur les données d’entraînement
2. **Reproductibilité** : random_state=42
3. **Normalisation obligatoire** pour la Régression Logistique
4. **Encodage One-Hot** pour éviter la multicolinéarité
5. **Équilibre des classes** assuré via la médiane

## 🤝 Contribution

1. Fork le projet
2. Crée une branche: `git checkout -b feature/improvement`
3. Commit: `git commit -m "Add improvement"`
4. Push: `git push origin feature/improvement`
5. Ouvre une Pull Request

