def should_promote(mae, best_mae):
   """Retourne True si le nouveau modele est meilleur."""
   return best_mae is not None and mae < best_mae
