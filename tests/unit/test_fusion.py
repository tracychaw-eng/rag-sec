from sec_rag.retrieval.fusion import rrf


def test_rrf_prefers_items_in_both_lists():
    scores = rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert scores["b"] > scores["c"]
    assert scores["a"] > scores["c"]
    # b: 1/62 + 1/61 > a: 1/61 + 1/63
    assert scores["b"] > scores["a"]


def test_rrf_single_list_preserves_order():
    scores = rrf([["x", "y", "z"]], k=60)
    assert scores["x"] > scores["y"] > scores["z"]


def test_rrf_empty():
    assert rrf([]) == {}
    assert rrf([[]]) == {}
