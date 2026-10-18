import pytest

from pl_predictor.sentiment import SentimentResult, SentimentScorer, preprocess


def test_preprocess_masks_mentions_and_urls():
    text = "@Arsenal what a win https://t.co/abc123 cc @friend_1"
    assert preprocess(text) == "@user what a win http cc @user"


def test_result_label_and_score():
    r = SentimentResult(negative=0.1, neutral=0.2, positive=0.7)
    assert r.label == "positive"
    assert r.score == pytest.approx(0.6)


@pytest.mark.parametrize(
    "output",
    [
        [
            {"label": "negative", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "positive", "score": 0.05},
        ],
        [
            {"label": "LABEL_0", "score": 0.8},
            {"label": "LABEL_1", "score": 0.15},
            {"label": "LABEL_2", "score": 0.05},
        ],
    ],
)
def test_from_pipeline_output(output):
    r = SentimentResult.from_pipeline_output(output)
    assert r.label == "negative"
    assert r.score == pytest.approx(-0.75)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        SentimentResult.from_pipeline_output([{"label": "joy", "score": 1.0}])


def test_scorer_uses_injected_pipeline():
    calls = {}

    def fake_pipeline(texts, **kwargs):
        calls["texts"], calls["kwargs"] = texts, kwargs
        return [
            [
                {"label": "positive", "score": 0.9},
                {"label": "neutral", "score": 0.05},
                {"label": "negative", "score": 0.05},
            ]
            for _ in texts
        ]

    scorer = SentimentScorer(pipeline=fake_pipeline, batch_size=4)
    results = scorer.score(["@LFC up the reds", "great game"])
    assert [r.label for r in results] == ["positive", "positive"]
    assert calls["texts"][0] == "@user up the reds"
    assert calls["kwargs"]["batch_size"] == 4
    assert scorer.score([]) == []
