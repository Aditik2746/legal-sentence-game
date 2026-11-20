
from app.core import milestone_1, milestone_2, milestone_4, milestone_5, predict_next


def test_milestone_1():
    corpus = milestone_1(["The Tenant shall PAY rent!"])
    assert corpus == [["the", "tenant", "shall", "pay", "rent"]]


def test_milestone_2():
    corpus = [["the", "tenant", "shall", "pay", "rent"]]
    model = milestone_2(corpus)
    assert model[("pay",)]["rent"] == 1


def test_milestone_3():
    from unittest.mock import patch

    with patch("app.classifier.classify", return_value="Payment"):
        from app.core import milestone_3
        assert milestone_3("The tenant shall pay rent.") == "Payment"


def test_milestone_4():
    sentences = milestone_4()
    assert len(sentences) == 5
    assert "The parties agree to maintain" in sentences


def test_milestone_5():
    from unittest.mock import patch
    with patch("app.core.milestone_3", return_value="Confidentiality"):
        result = milestone_5("The parties agree to maintain confidentiality.")
    assert result["category"] == "Confidentiality"
    assert "Confidentiality" in result["feedback"]
