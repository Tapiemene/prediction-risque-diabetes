import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

# CONFIGURATION
FIGURE_SIZE = (10, 6)
LARGE_FIGURE_SIZE = (15, 10)
SMALL_FIGURE_SIZE = (6, 5)
HISTOGRAM_BINS = 30
COLOR_PALETTE = 'coolwarm'
SAVE_PLOTS = True
PLOT_FORMAT = 'png'
PLOT_DPI = 300


def initialize_plots() -> int:
    """
    Initialise l’environnement de visualisation et retourne le compteur de graphiques.
    
    Retour :
    --------
    int
        Numéro de départ des graphiques (généralement 1)
    """
    
    # Définir le style seaborn
    sns.set_style("darkgrid")
    
    # Définir la palette de couleurs
    sns.set_palette(COLOR_PALETTE)
    
    # Supprimer certains warnings matplotlib
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    return 1


def show_fig(plot_no: int) -> None:
    """
    Affiche la figure actuelle avec un layout optimisé.
    
    Paramètres :
    ------------
    plot_no : int
        Numéro du graphique (pour affichage)
    """
    
    plt.tight_layout()
    plt.show()


def save_fig(filename: str) -> None:
    """
    Sauvegarde la figure actuelle sur disque.
    
    Paramètres :
    ------------
    filename : str
        Nom du fichier de sortie (sans extension)
    
    Sauvegarde dans : results/{filename}.png (300 DPI)
    """
    
    if SAVE_PLOTS:
        filepath = f'results/{filename}.{PLOT_FORMAT}'
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"   Saved: {filepath}")


# ANALYSE DES DISTRIBUTIONS

def plot_age_distribution(df: pd.DataFrame, plot_no: int) -> int:
    """
    Histogramme de distribution des âges avec courbe KDE.
        
    Objectifs :
    - Montrer la distribution des âges
    - Vérifier si la distribution est normale
    - Détecter les valeurs extrêmes
    """
    
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    sns.histplot(df['age'], kde=True, bins=HISTOGRAM_BINS)
    
    plt.title(f'{plot_no}. Distribution des âges et vulnérabilité potentielle au diabète')
    plt.xlabel('Âge (années)')
    plt.ylabel('Fréquence')
    
    show_fig(plot_no)
    save_fig(f'plot_{plot_no:02d}_age_distribution')
    
    return plot_no + 1


def plot_risk_category_distribution(df: pd.DataFrame, plot_no: int) -> int:
    """
    Distribution des catégories de risque de diabète.
    
    Type de visualisation : Diagramme en barres (countplot)
    
    Objectifs :
    - Visualiser la répartition des patients par catégorie de risque
    - Détecter un déséquilibre des classes
    - Comprendre la structure du dataset
    """
    
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    sns.countplot(
        x='diabetes_risk_category',
        data=df,
        order=df['diabetes_risk_category'].value_counts().index
    )
    
    plt.title(f'{plot_no}. Répartition des patients selon les catégories de risque')
    plt.xlabel('Catégorie de risque')
    plt.ylabel('Nombre de patients')
    
    show_fig(plot_no)
    save_fig(f'plot_{plot_no:02d}_risk_distribution')
    
    return plot_no + 1


# ANALYSE DES RELATIONS ENTRE VARIABLES

def plot_bmi_by_risk(df: pd.DataFrame, plot_no: int) -> int:
    """
    Variation de l’IMC selon les catégories de risque.
        
    Objectifs :
    - Comparer les distributions d’IMC entre catégories
    - Visualiser médiane, quartiles et outliers
    - Vérifier si l’IMC est un bon prédicteur
    """
    
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    sns.boxplot(x='diabetes_risk_category', y='bmi', data=df)
    
    plt.title(f'{plot_no}. Variation de l’IMC selon les catégories de risque')
    plt.xlabel('Catégorie de risque')
    plt.ylabel('IMC')
    
    show_fig(plot_no)
    save_fig(f'plot_{plot_no:02d}_bmi_by_risk')
    
    return plot_no + 1


def plot_glucose_vs_hba1c(df: pd.DataFrame, plot_no: int) -> int:
    """
    Relation entre glycémie à jeun et HbA1c selon la catégorie de risque.
        
    Objectifs :
    - Visualiser la corrélation glucose ↔ HbA1c
    - Colorer selon la catégorie de risque
    - Détecter les anomalies
    """
    
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    sns.scatterplot(
        x='fasting_glucose_level',
        y='HbA1c_level',
        hue='diabetes_risk_category',
        data=df
    )
    
    plt.title(f'{plot_no}. Relation entre glycémie à jeun et HbA1c selon le risque')
    plt.xlabel('Glycémie à jeun (mg/dL)')
    plt.ylabel('HbA1c (%)')
    
    show_fig(plot_no)
    save_fig(f'plot_{plot_no:02d}_glucose_vs_hba1c')
    
    return plot_no + 1


def plot_correlation_heatmap(df: pd.DataFrame, plot_no: int) -> int:
    """
    Heatmap de corrélation entre toutes les variables numériques.
    
    Objectifs :
    - Visualiser les corrélations entre features
    - Détecter la multicolinéarité
    - Identifier les meilleurs prédicteurs
    
    Interprétation :
    - Rouge : forte corrélation positive
    - Bleu : forte corrélation négative
    - Clair : faible corrélation
    """
    
    fig = plt.figure(figsize=LARGE_FIGURE_SIZE)
    
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    
    plt.title(f'{plot_no}. Matrice de corrélation des variables cliniques et de mode de vie')
    
    show_fig(plot_no)
    save_fig(f'plot_{plot_no:02d}_correlation_heatmap')
    
    return plot_no + 1


# ANALYSE DES VARIABLES CATÉGORIELLES

def plot_gender_by_risk(df: pd.DataFrame, plot_no: int) -> int:
    """
    Distribution du score de risque selon le genre.
        
    Objectifs :
    - Comparer les distributions entre genres
    - Visualiser la forme complète de la distribution
    """
    
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    sns.violinplot(x='gender', y='diabetes_risk_score', data=df)
    
    plt.title(f'{plot_no}. Distribution du score de risque selon le genre')
    plt.xlabel('Genre')
    plt.ylabel('Score de risque')
    
    show_fig(plot_no)
    save_fig(f'plot_{plot_no:02d}_gender_by_risk')
    
    return plot_no + 1


def plot_family_history_impact(df: pd.DataFrame, plot_no: int) -> int:
    """
    Impact des antécédents familiaux sur le score de risque.
        
    Objectifs :
    - Comparer les scores entre patients avec/sans antécédents
    - Évaluer l’influence génétique
    """
    
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    sns.boxplot(x='family_history_diabetes', y='diabetes_risk_score', data=df)
    
    plt.title(f'{plot_no}. Impact des antécédents familiaux sur le score de risque')
    plt.xlabel('Antécédents familiaux de diabète')
    plt.ylabel('Score de risque')
    
    show_fig(plot_no)
    save_fig(f'plot_{plot_no:02d}_family_history_impact')
    
    return plot_no + 1


# FONCTION EDA COMPLÈTE

def perform_eda(df: pd.DataFrame) -> None:
    """
    Exécute l’analyse exploratoire complète (EDA).
    
    Génère plus de 20 visualisations.
    
    Pipeline :
    ----------
    1. Initialisation
    2. Graphiques de distribution
    3. Graphiques de relations
    4. Graphiques catégoriels
    5. Comparaisons de features
    
    Sorties :
    ---------
    - Affiche ou sauvegarde les graphiques
    - Imprime les insights clés
    """
    
    print("=" * 80)
    print("DÉBUT DE L’ANALYSE EXPLORATOIRE (EDA)")
    print("=" * 80 + "\n")
    
    plot_no = initialize_plots()
    
    print("ANALYSE DES DISTRIBUTIONS")
    plot_no = plot_age_distribution(df, plot_no)
    plot_no = plot_risk_category_distribution(df, plot_no)
    
    print("\nANALYSE DES RELATIONS")
    plot_no = plot_bmi_by_risk(df, plot_no)
    plot_no = plot_glucose_vs_hba1c(df, plot_no)
    plot_no = plot_correlation_heatmap(df, plot_no)
    
    print("\nANALYSE CATÉGORIELLE")
    plot_no = plot_gender_by_risk(df, plot_no)
    plot_no = plot_family_history_impact(df, plot_no)
    
    print("\n" + "=" * 80)
    print("ANALYSE EXPLORATOIRE TERMINÉE")
    print(f"{plot_no - 1} visualisations générées")
    print("=" * 80 + "\n")
    
    print_eda_insights(df)


def print_eda_insights(df: pd.DataFrame) -> None:
    """
    Affiche les insights clés issus de l’EDA.
    
    Affiche :
    - Répartition des classes
    - Corrélations principales
    - Valeurs manquantes
    - Statistiques importantes
    """
    
    print("INSIGHTS EDA :\n")
    
    risk_counts = df['diabetes_risk_category'].value_counts()
    print("Répartition des catégories de risque :")
    for cat, count in risk_counts.items():
        pct = count / len(df) * 100
        print(f"  • {cat} : {count} patients ({pct:.1f}%)")
    
    print("\nPrincipales corrélations avec le score de risque :")
    corr_with_target = (
        df.select_dtypes(include=['int64', 'float64'])
        .corr()['diabetes_risk_score']
        .sort_values(ascending=False)[1:6]
    )
    for feature, corr in corr_with_target.items():
        print(f"  • {feature} : {corr:.3f}")
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("\nValeurs manquantes :")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  • {col} : {count}")
    else:
        print("\nAucune valeur manquante détectée")
    
    print()
