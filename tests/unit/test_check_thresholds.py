from eval.check_thresholds import check

THRESHOLDS = {
    "ragas": {"faithfulness": 0.85, "answer_relevancy": 0.75},
    "retrieval": {"source_recall@K": 0.95},
    "abstention_rate": 0.8,
}


def _metrics(faith=0.9, relev=0.8, recall=1.0, abstain=5):
    return {
        "ragas": {"overall": {"faithfulness": faith,
                              "answer_relevancy": relev}},
        "retrieval": {"overall": {"source_recall@K": recall}},
        "abstention": {f"A{i}": i < abstain for i in range(5)},
    }


def test_all_above_floors_passes():
    assert check(_metrics(), THRESHOLDS) == []


def test_ragas_below_floor_fails():
    v = check(_metrics(faith=0.7), THRESHOLDS)
    assert any("faithfulness" in x for x in v)


def test_retrieval_below_floor_fails():
    v = check(_metrics(recall=0.5), THRESHOLDS)
    assert any("source_recall" in x for x in v)


def test_abstention_rate_fails():
    v = check(_metrics(abstain=2), THRESHOLDS)
    assert any("abstention_rate" in x for x in v)


def test_missing_metric_reported():
    v = check({"ragas": {"overall": {}}}, THRESHOLDS)
    assert any("missing" in x for x in v)


def test_dataset_version_mismatch_fails():
    thresholds = {**THRESHOLDS, "dataset_version": 4}
    m = _metrics()
    m["dataset_version"] = 3
    v = check(m, thresholds)
    assert any("dataset_version mismatch" in x for x in v)


def test_matching_dataset_version_passes():
    thresholds = {**THRESHOLDS, "dataset_version": 4}
    m = _metrics()
    m["dataset_version"] = 4
    assert check(m, thresholds) == []


def test_legacy_report_without_version_not_blocked():
    # pre-backfill reports lack the field — absence is not a violation
    thresholds = {**THRESHOLDS, "dataset_version": 4}
    assert check(_metrics(), thresholds) == []
