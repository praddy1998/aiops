from src.features import select_training_cols_columns

def test_select_cols():
    cols = set(select_training_cols_columns())
    assert cols == {"trip_distance","passenger_count","pickup_hour","pickup_dow","label"}
