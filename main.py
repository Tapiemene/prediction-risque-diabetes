import sys
import os
from pathlib import Path

# Ajouter le dossier src au PATH pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# IMPORT DES MODULES DU PIPELINE
from src.data_loader import load_dataset, validate_dataset, print_dataset_info
from src.eda import perform_eda
from src.preprocessing import preprocess_data
from src.model import train_model, generate_full_evaluation

# CONSTANTES
DATASET_PATH = 'data/diabetes_risk_dataset.csv'  # Chemin vers le fichier CSV

def print_header(title: str) -> None:
    """
    Affiche un en-tête de section formaté.
    
    Paramètres :
    ------------
    title : str
        Le texte du titre à afficher
    
    Sortie :
    --------
    Affiche un en-tête formaté avec des lignes décoratives
    """
    
    border = "=" * 80
    print(f"\n{border}")
    print(f"{title:^80}")
    print(f"{border}\n")


def main():
    """
    Exécute le pipeline complet du chargement des données à l’évaluation du modèle.
    
    Sorties :
    ---------
    Console :
    - Messages de progression
    - Statistiques et métriques
    - Rapport d’évaluation détaillé
    """

    try:
        # ÉTAPE 1 : CHARGEMENT DES DONNÉES
        print_header("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
        
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                f"Fichier dataset introuvable : {DATASET_PATH}\n"
                f"Veuillez vérifier que le fichier se trouve bien dans le dossier data/"
            )
        
        # Charger le dataset
        df = load_dataset(DATASET_PATH)
        
        # Valider le dataset
        is_valid, warnings = validate_dataset(df)
        if warnings:
            print("!! Avertissements de validation :")
            for warning in warnings:
                print(f"   {warning}")
        
        # Afficher les informations du dataset
        print_dataset_info(df)
        
        # ÉTAPE 2 : ANALYSE EXPLORATOIRE
        print_header("ÉTAPE 2 : ANALYSE EXPLORATOIRE (EDA)")
        perform_eda(df)
        
        # ÉTAPE 3 : PRÉTRAITEMENT
        print_header("ÉTAPE 3 : PRÉTRAITEMENT DES DONNÉES")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE  
        print_header("ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE")
        model = train_model(X_train, y_train)
        
        # ÉTAPE 5 : ÉVALUATION DU MODÈLE
        print_header("ÉTAPE 5 : ÉVALUATION DU MODÈLE")
        results = generate_full_evaluation(model, X_test, y_test)
        
        # PIPELINE TERMINÉ
        print_header("PIPELINE TERMINÉ")
        
        print(f"Résumé des résultats :")
        print(f"    Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"    Données d’entraînement : {X_train.shape[0]}")
        print(f"    Données de test : {X_test.shape[0]}")
        print(f"    Accuracy finale : {results['accuracy']*100:.2f}%")
        print(f"    F1-Score : {results['f1_score']:.4f}")
        print(f"\nModèle entraîné et évalué avec succès ;)\n")
        
        return model, results, (X_train, X_test, y_train, y_test)
    
    # GESTION DES ERREURS
    except FileNotFoundError as e:
        print(f"\n!! ERREUR : {e}\n")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n!! ERREUR DE DONNÉES : {e}\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n!! ERREUR INATTENDUE : {e}\n")
        print("Veuillez vérifier vos données et réessayer.")
        sys.exit(1)

# POINT D’ENTRÉE

if __name__ == "__main__":
    """
    Exécute le pipeline principal lorsque le script est lancé directement.
    """
    
    model, results, data = main()  
