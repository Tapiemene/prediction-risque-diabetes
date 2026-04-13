"""
Module de Prétraitement des Données

Gère l’ingénierie des features, l’encodage, la normalisation et la séparation
train/test. Garantit l’absence de fuite de données entre train et test.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

# CONFIGURATION
DROP_COLUMNS = ['Patient_ID', 'diabetes_risk_score']
TARGET_COLUMN = 'diabetes_risk_score'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def create_target_variable(df: pd.DataFrame, target_col: str = 'diabetes_risk_score') -> np.ndarray:
    """
    Convertit le score continu diabetes_risk_score en classification binaire.
    
    Paramètres :
    ------------
    df : pd.DataFrame
        DataFrame contenant la colonne cible
    target_col : str
        Nom de la colonne cible
    
    Retour :
    --------
    np.ndarray
        Tableau binaire :
        - 1 = Haut risque (score > médiane)
        - 0 = Faible risque (score ≤ médiane)
    """
    
    scores = df[target_col]
    median_score = scores.median()
    y = (scores > median_score).astype(int)
    
    print(f"Variable cible créée :")
    print(f"   Score médian : {median_score:.2f}")
    print(f"   Classe 0 (Faible risque) : {(y == 0).sum()} échantillons")
    print(f"   Classe 1 (Haut risque)   : {(y == 1).sum()} échantillons")
    print(f"   Balance : {(y == 1).sum() / len(y) * 100:.1f}% Haut risque\n")
    
    return y


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les variables catégorielles en colonnes numériques via one-hot encoding.
    
    Retour :
    --------
    pd.DataFrame
        Matrice de features avec colonnes catégorielles encodées.
    """
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"Encodage One-Hot appliqué :")
    print(f"   Features d’origine : {X.shape[1]}")
    print(f"   Features encodées : {X_encoded.shape[1]}\n")
    
    return X_encoded


def split_train_test(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Sépare les données en ensembles d’entraînement et de test.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Séparation Train/Test effectuée :")
    print(f"   Train : {X_train.shape[0]} échantillons ({(1-test_size)*100:.0f}%)")
    print(f"   Test  : {X_test.shape[0]} échantillons ({test_size*100:.0f}%)")
    print(f"   Nombre de features : {X_train.shape[1]}")
    print(f"   Random state : {random_state}\n")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalise les features via StandardScaler.
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Normalisation appliquée (StandardScaler) :")
    print(f"   Train - Mean : {X_train_scaled.mean():.6f}, Std : {X_train_scaled.std():.6f}")
    print(f"   Test  - Mean : {X_test_scaled.mean():.6f}, Std : {X_test_scaled.std():.6f}")
    print(f"   (Différences normales dues à la taille du test)\n")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Pipeline complet de prétraitement.
    """
    
    print("=" * 80)
    print("DÉBUT DU PIPELINE DE PRÉTRAITEMENT")
    print("=" * 80 + "\n")
    
    # Étape 1 : suppression des colonnes inutiles
    X = df.drop(columns=DROP_COLUMNS)
    print(f"1️ Colonnes supprimées : {DROP_COLUMNS}")
    print(f"   Features restants : {X.shape[1]}\n")
    
    # Étape 2 : création de la cible binaire
    y = create_target_variable(df, TARGET_COLUMN)
    
    # Étape 3 : encodage des variables catégorielles
    X = encode_categorical_features(X)
    
    # Étape 4 : séparation train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Étape 5 : normalisation
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("=" * 80)
    print("PIPELINE DE PRÉTRAITEMENT TERMINÉ ")
    print("=" * 80 + "\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
