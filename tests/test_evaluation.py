"""Testes da avaliação de métricas do modelo."""

import unittest
import contextlib
import io

import torch

from src.evaluation.metrics import evaluate_model


class TestEvaluateModel(unittest.TestCase):
    def test_output_contains_all_metric_names(self) -> None:
        y = torch.tensor([1, 0, 1, 0])
        y_pred = torch.tensor([1, 0, 1, 0])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            evaluate_model("cpu", y, y_pred)
        text = output.getvalue()
        self.assertIn("Accuracy", text)
        self.assertIn("Precision", text)
        self.assertIn("Recall", text)
        self.assertIn("F1-Score", text)

    def test_perfect_predictions_accuracy_is_one(self) -> None:
        y = torch.tensor([1, 0, 1, 0, 1])
        y_pred = torch.tensor([1, 0, 1, 0, 1])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            evaluate_model("cpu", y, y_pred)
        self.assertIn("Accuracy: 1.0000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
