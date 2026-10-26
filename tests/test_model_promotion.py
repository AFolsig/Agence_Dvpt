from models.model_promotion import should_promote


def test_better_model_is_promoted():
   assert should_promote(1200, 1500) is True


def test_worse_model_is_not_promoted():
   assert should_promote(1600, 1500) is False


def test_equal_performance_is_not_promoted():
   assert should_promote(1500, 1500) is False


def test_missing_champion_mae():
   assert should_promote(1200, None) is False
