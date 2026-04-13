"""
Ce module gère le chargement du dataset de diabète depuis un fichier CSV.
Il inclut la validation et des vérifications basiques d’intégrité des données.

"""
import pandas as pd
import os
from typing import Tuple

# IMPORT CONFIG
from src.config import DATASET_PATH
# NOTE : Dans l’implémentation réelle, importer la config ici
# Pour l’instant, utilisation de chemins relatifs

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Charger le dataset de diabète depuis un fichier CSV.
    
    Paramètres :
    ------------
    filepath : str
        Chemin vers le fichier CSV contenant les données.
        Peut être un chemin absolu ou relatif.
    
    Retourne :
    ----------
    pd.DataFrame
        Données chargées sous forme de DataFrame pandas
        avec la forme (n_samples, n_features)
    """
    
    # Vérifier si le fichier existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset introuvable à l’emplacement : {filepath}\n"
            f"Veuillez le télécharger depuis : "
            f"https://www.kaggle.com/datasets/vishardmehta/diabetes-risk-prediction-dataset/data\n"
            f"Et placer le fichier CSV dans le dossier data/."
        )
    
    # Charger le fichier CSV dans un DataFrame
    # =========================================================================
    # pd.read_csv() - Lecture d’un fichier CSV
    # Paramètres :
    #   filepath : chemin du fichier CSV
    #   index_col=None : ne définit aucune colonne comme index
    #   dtype=None : laisse pandas deviner les types
    # Retour : DataFrame pandas de forme (n_lignes, n_colonnes)
    # =========================================================================
    df = pd.read_csv(filepath)
    
    # Vérifier que les données ne sont pas vides
    if df.empty:
        raise ValueError(f"Le dataset à {filepath} est vide !")
    
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Valider la structure et le contenu du dataset.
    
    Paramètres :
    ------------
    df : pd.DataFrame
        Dataset chargé à valider
    
    Retourne :
    ----------
    Tuple[bool, list]
        (is_valid, liste_d_avertissements)
        - is_valid : True si aucun problème critique n’est détecté
        - warnings : liste des avertissements non critiques
    
    Exemple :
    ---------
    >>> is_valid, warnings = validate_dataset(df)
    >>> if not is_valid:
    ...     for w in warnings:
    ...         print(f"⚠️  {w}")
    """
    
    warnings = []
    
    # Vérifier que les colonnes obligatoires existent
    required_cols = [
        'Patient_ID', 'age', 'bmi', 'gender',
        'fasting_glucose_level', 'HbA1c_level',
        'diabetes_risk_score'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.append(f"Colonnes manquantes : {missing_cols}")
        return False, warnings
    
    # Vérifier si le dataset est anormalement petit
    if len(df) < 100:
        warnings.append(f"Dataset très petit : seulement {len(df)} lignes")
    
    # Vérifier les valeurs nulles
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        null_info = null_counts[null_counts > 0]
        warnings.append(f"Valeurs nulles détectées :\n{null_info}")
    
    # Vérifier les doublons
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(f"{dup_count} lignes dupliquées détectées")
    
    return True, warnings


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Afficher les informations de base du dataset.
    
    Paramètres :
    ------------
    df : pd.DataFrame
        Dataset chargé
    
    Sortie :
    --------
    Affiche dans la console :
    - Forme du dataset (lignes, colonnes)
    - Noms et types des colonnes
    - Nombre de valeurs non nulles
    - Utilisation mémoire
    """
    
    print("=" * 80)
    print("INFORMATIONS SUR LE DATASET")
    print("=" * 80)
    
    # Forme du dataset : (nombre de lignes, nombre de colonnes)
    print(f"\n📊 Forme du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    # Types des colonnes
    print("\n📋 Types des colonnes :")
    print(df.dtypes)
    
    # Informations détaillées
    print("\n📌 Informations détaillées :")
    df.info()
    
    # Résumé statistique des colonnes numériques
    print("\n📈 Résumé statistique :")
    print(df.describe())


# ============================================================================
# INTÉGRATION AU PIPELINE
# ============================================================================
# ÉTAPE SUIVANTE : Une fois les données chargées, passer à src/preprocessing.py :
#
# from src.data_loader import load_dataset
# from src.preprocessing import preprocess_data
#
# df = load_dataset('data/diabetes_risk_dataset.csv')
# X, y = preprocess_data(df)
# ============================================================================


if __name__ == "__main__":
    # Exemple d’utilisation
    dataset_path = 'data/diabetes_risk_dataset.csv'
    
    # Charger les données
    df = load_dataset(dataset_path)
    
    # Valider les données
    is_valid, warnings = validate_dataset(df)
    if warnings:
        print("\n!! Avertissements de validation :")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Afficher les infos
    print_dataset_info(df)
    
    print("\n✅ Données chargées avec succès !")
