"""
Module d’Entraînement et d’Évaluation du Modèle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from typing import Tuple

# CONFIGURATION
MODEL_MAX_ITER = 2000
MODEL_C = 1.0
MODEL_SOLVER = 'lbfgs'

def train_model(X_train: np.ndarray, y_train: np.ndarray, max_iter: int = MODEL_MAX_ITER, C: float = MODEL_C,
                solver: str = MODEL_SOLVER) -> LogisticRegression:
    """
    Entraîne un modèle de Régression Logistique sur les données d’entraînement.
    
    Paramètres :
    ------------
    X_train : np.ndarray
        Features d’entraînement, forme (n_samples, n_features)
    y_train : np.ndarray
        Cibles d’entraînement (binaire : 0 ou 1)
    max_iter : int
        Nombre maximal d’itérations pour la convergence
    C : float
        Inverse de la force de régularisation
    solver : str
        Algorithme d’optimisation
    
    Retour :
    --------
    LogisticRegression
        Modèle entraîné avec coefficients appris
    """
    
    print("=" * 80)
    print("DÉBUT DE L’ENTRAÎNEMENT DU MODÈLE")
    print("=" * 80 + "\n")
    
    print(f"Entraînement du modèle de Régression Logistique")
    print(f"   Solveur : {solver}")
    print(f"   Itérations max : {max_iter}")
    print(f"   Régularisation (C) : {C}")
    print(f"   Échantillons d’entraînement : {X_train.shape[0]}")
    print(f"   Nombre de features : {X_train.shape[1]}")
    print(f"   Classes : Binaire (0, 1)\n")
    
    model = LogisticRegression(max_iter=max_iter, C=C, solver=solver)
    model.fit(X_train, y_train)
    
    print(f"Entraînement terminé !\n")
    print(f"   Nombre de coefficients appris : {model.n_features_in_}")
    print(f"   Intercept (biais) : {model.intercept_[0]:.6f}")
    print(f"   Magnitude moyenne des coefficients : {np.abs(model.coef_[0]).mean():.6f}\n")
    
    return model


def make_predictions(model: LogisticRegression, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère les prédictions sur l’ensemble de test.
    
    Paramètres :
    ------------
    model : LogisticRegression
        Modèle entraîné
    X_test : np.ndarray
        Features de test
    
    Retour :
    --------
    y_pred : np.ndarray
        Prédictions (0 ou 1)
    y_prob : np.ndarray
        Probabilités prédites (0.0 – 1.0)
    """
    
    print("Génération des prédictions\n")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"   Prédictions générées pour {len(y_pred)} échantillons")
    print(f"   Classe 0 (Faible risque) : {(y_pred == 0).sum()} ({(y_pred == 0).sum()/len(y_pred)*100:.1f}%)")
    print(f"   Classe 1 (Haut risque)   : {(y_pred == 1).sum()} ({(y_pred == 1).sum()/len(y_pred)*100:.1f}%)")
    print(f"   Probabilité moyenne : {y_prob.mean():.3f}")
    print(f"   Intervalle des probabilités : [{y_prob.min():.3f}, {y_prob.max():.3f}]\n")
    
    return y_pred, y_prob


def evaluate_model(model: LogisticRegression,
                  X_test: np.ndarray,
                  y_test: np.ndarray) -> Tuple[float, dict]:
    """
    Évalue les performances du modèle sur l’ensemble de test.
    
    Métriques calculées :
    ---------------------
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Rapport de classification
    """
    
    print("=" * 80)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 80 + "\n")
    
    y_pred, y_prob = make_predictions(model, X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print("MÉTRIQUES DE PERFORMANCE :\n")
    print(f"   Accuracy :  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall :    {recall:.4f}")
    print(f"   F1-score :  {f1:.4f}\n")
    
    print("RAPPORT DE CLASSIFICATION :\n")
    report = classification_report(y_test, y_pred, target_names=['Faible risque (0)', 'Haut risque (1)'])
    print(report)
    
    return accuracy, metrics


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, title: str = "Matrice de Confusion") -> None:
    """
    Affiche la matrice de confusion sous forme de heatmap.
    """
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title(title)
    plt.xlabel("Prédiction")
    plt.ylabel("Valeur réelle")
    plt.xticks([0.5, 1.5], ['Faible risque (0)', 'Haut risque (1)'])
    plt.yticks([0.5, 1.5], ['Faible risque (0)', 'Haut risque (1)'])
    
    tn, fp, fn, tp = cm.ravel()
    print("\nDÉTAIL DE LA MATRICE DE CONFUSION :\n")
    print(f"   Vrais négatifs (TN) : {tn}")
    print(f"   Faux positifs (FP)  : {fp}")
    print(f"   Faux négatifs (FN)  : {fn}")
    print(f"   Vrais positifs (TP) : {tp}")
    print(f"   Taux d’erreur : {(fp+fn)*100/len(y_test):.2f}%\n")
    
    plt.tight_layout()
    plt.show()


def generate_full_evaluation(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Pipeline complet d’évaluation : métriques + visualisations.
    
    Retour :
    --------
    dict contenant toutes les métriques et prédictions
    """
    
    accuracy, metrics = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, metrics['y_pred'])
    
    print("=" * 80)
    print("ÉVALUATION DU MODÈLE TERMINÉE")
    print("=" * 80 + "\n")
    
    return metrics
