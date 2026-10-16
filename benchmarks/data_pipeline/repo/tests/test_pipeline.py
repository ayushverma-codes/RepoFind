from pipeline import collect


def test_collect_adds_item():
    assert collect(1, []) == [1]
