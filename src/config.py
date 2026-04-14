import os

# CONFIGURATION DES CHEMINS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Nom du fichier du dataset
DATASET_PATH = os.path.join(DATA_DIR, 'diabetes_risk_dataset.csv')

# CONFIGURATION DU TRAITEMENT DES DONNÉES

# Ratio de séparation train/test
# 20% pour le test, 80% pour l'entraînement
TEST_SIZE = 0.2

# Graine aléatoire pour la reproductibilité
RANDOM_STATE = 42

# Colonnes à supprimer des features (Patient_ID n’a aucune valeur prédictive)
DROP_COLUMNS = ['Patient_ID', 'diabetes_risk_score']

# Colonne cible à prédire
TARGET_COLUMN = 'diabetes_risk_score'

# CONFIGURATION DU MODÈLE

# Hyperparamètres de la régression logistique
MODEL_MAX_ITER = 2000  # max_iter : nombre maximal d’itérations pour la convergence du solveur

# Paramètre de régularisation du modèle
MODEL_C = 1.0

# Algorithme du solveur
MODEL_SOLVER = 'lbfgs'

# CONFIGURATION DES VISUALISATIONS

# Taille par défaut des figures (largeur, hauteur) en pouces
FIGURE_SIZE = (10, 6)

# Grande taille de figure pour la heatmap de corrélation
LARGE_FIGURE_SIZE = (15, 10)

# Petite taille de figure pour la matrice de confusion
SMALL_FIGURE_SIZE = (6, 5)

# Nombre de bins pour les histogrammes
HISTOGRAM_BINS = 30

# Palette de couleurs pour les visualisations
COLOR_PALETTE = 'coolwarm'

# Style des graphiques
PLOT_STYLE = 'darkgrid'

# CONFIGURATION EDA (Analyse Exploratoire)

# Activer/désactiver certaines étapes d’analyse
PERFORM_EDA = True
PERFORM_CORRELATION_ANALYSIS = True
PERFORM_DISTRIBUTION_ANALYSIS = True

# Nombre de points pour l’estimation de densité (KDE)
KDE_POINTS = 30

# CONFIGURATION DES SORTIES

# Sauvegarder les visualisations sur disque
SAVE_PLOTS = True

# Format des images sauvegardées (png, pdf, svg, etc.)
PLOT_FORMAT = 'png'

# DPI des images (résolution)
PLOT_DPI = 150

# Afficher les métriques détaillées du modèle
VERBOSE = True

# CONFIGURATION DES FEATURES

# Features numériques (utilisées pour le scaling et l’analyse de corrélation)
NUMERICAL_FEATURES = [
    'age',
    'bmi',
    'waist_circumference_cm',
    'fasting_glucose_level',
    'HbA1c_level',
    'blood_pressure',
    'insulin_level',
    'triglycerides_level',
    'cholesterol_level',
    'diabetes_risk_score',
    'sleep_hours',
    'stress_level',
    'daily_calorie_intake',
    'sugar_intake_grams_per_day'
]

# Features catégorielles (seront encodées en one-hot)
CATEGORICAL_FEATURES = [
    'gender',
    'family_history_diabetes',
    'physical_activity_level'
]

# CONFIGURATION DU LOGGING

# Niveau de logging
LOG_LEVEL = 'INFO'  # Options : DEBUG, INFO, WARNING, ERROR, CRITICAL

# Fichier de log
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'diabetes_model.log')

# Créer le dossier logs s’il n’existe pas
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
