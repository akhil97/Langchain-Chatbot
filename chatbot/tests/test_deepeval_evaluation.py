import deepeval.metrics as deepeval_metrics
import deepeval.models as deepeval_models

from rag_services import evaluate_answer_with_deepeval


class FakeJudge:
    def __init__(self, **values):
        self.values = values


class FakeMetric:
    instances = []

    def __init__(self, **values):
        self.values = values
        self.score = None
        self.reason = ""
        self.instances.append(self)

    def measure(self, test_case):
        self.test_case = test_case
        self.score = 0.82
        self.reason = "Evaluation passed."


def patch_deepeval(monkeypatch):
    FakeMetric.instances.clear()
    monkeypatch.setattr(deepeval_models, "OpenRouterModel", FakeJudge)
    for metric_name in (
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "ContextualRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
    ):
        monkeypatch.setattr(deepeval_metrics, metric_name, FakeMetric)


def test_evaluates_answer_and_retrieval_with_reference(monkeypatch):
    patch_deepeval(monkeypatch)

    results = evaluate_answer_with_deepeval(
        question="What does the course cover?",
        answer="It covers data engineering.",
        contexts=["The course covers data engineering."],
        reference="Data engineering.",
        api_key="test-key",
        model_name="openai/gpt-4o-mini",
    )

    assert set(results) == {
        "answer_relevancy",
        "faithfulness",
        "contextual_relevancy",
        "contextual_precision",
        "contextual_recall",
    }
    assert len(FakeMetric.instances) == 5
    assert all(metric.values["threshold"] == 0.7 for metric in FakeMetric.instances)
    assert all(metric.values["async_mode"] is False for metric in FakeMetric.instances)
    assert all(
        result == {
            "score": 0.82,
            "reason": "Evaluation passed.",
            "passed": True,
        }
        for result in results.values()
    )
    assert (
        FakeMetric.instances[0].test_case.expected_output == "Data engineering."
    )


def test_runs_only_answer_relevancy_without_retrieval(monkeypatch):
    patch_deepeval(monkeypatch)

    results = evaluate_answer_with_deepeval(
        question="What is 25 times 4?",
        answer="100.",
        contexts=[],
        reference="100",
        api_key="test-key",
        model_name="openai/gpt-4o-mini",
    )

    assert set(results) == {"answer_relevancy"}
    assert len(FakeMetric.instances) == 1
    assert FakeMetric.instances[0].test_case.retrieval_context is None
