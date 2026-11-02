def should_promote(mae, best_mae, min_improvement=0.001):
    """
    Promouvoir le nouveau modèle uniquement si sa MAE
    s'améliore d'au moins 0,1 %.
    """
    if best_mae is None:
        return True

    return mae <= best_mae * (1 - min_improvement)
