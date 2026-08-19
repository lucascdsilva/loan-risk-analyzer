"""Testes do bundle de inferência (itens A2 e A6 de docs/PLANO_ARQUITETURA.md).

Verificam o lado do projeto de treino: que o grafo exportado reproduz o modelo
PyTorch, que o bundle é autocontido e que o `golden.json` publicado é
verdadeiro. O lado do consumidor (encoder reimplementado em numpy contra este
mesmo golden) é o `test_contract.py` do loan-risk-ml-service.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.export.bundle import (
    build_exportable,
    export_bundle,
    export_onnx,
    run_onnx,
)
from src.models.NeuralNetworkV0 import NeuralNetworkV0
from src.preprocessing.transform import (
    EDUCATION_ORDER,
    N_FEATURES,
    ONE_HOT_CATEGORIES,
    fit_encoder,
    fit_scaler,
    transform_records,
)

# Diferença esperada entre torch e onnxruntime em float32. Medida em ~1,5e-6
# sobre 500 registros; 1e-5 dá folga sem esconder um erro real de tradução.
PARITY_TOLERANCE = 1e-5

BUNDLE_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "bundle"


def _dataset(n_repeats: int = 6) -> pd.DataFrame:
    """Dataset sintético cobrindo todas as categorias do contrato."""
    educations = list(EDUCATION_ORDER)
    rows = []
    for r in range(n_repeats):
        for i, home in enumerate(ONE_HOT_CATEGORIES["person_home_ownership"]):
            for j, intent in enumerate(ONE_HOT_CATEGORIES["loan_intent"]):
                rows.append(dict(
                    person_age=25.0 + (i + j + r) % 40,
                    person_gender="female" if (i + j) % 2 else "male",
                    person_education=educations[(i + j + r) % len(educations)],
                    person_income=40000.0 + 900 * (i + 3 * j + 7 * r),
                    person_emp_exp=(i + j + r) % 25,
                    person_home_ownership=home,
                    loan_amnt=5000.0 + 700 * (j + r),
                    loan_intent=intent,
                    loan_int_rate=6.0 + (i + j) % 12,
                    loan_percent_income=0.05 + 0.01 * ((i + j + r) % 20),
                    cb_person_cred_hist_length=2.0 + (i + r) % 15,
                    credit_score=520 + 11 * ((i + 2 * j + r) % 30),
                    previous_loan_defaults_on_file="Yes" if (i + j + r) % 2 else "No",
                    loan_status=(i + j + r) % 2,
                ))
    return pd.DataFrame(rows)


def _fit_all(dataset: pd.DataFrame):
    encoder, X, y, feature_names = fit_encoder(dataset)
    scaler = fit_scaler(X)
    torch.manual_seed(0)
    model = NeuralNetworkV0(in_features=X.shape[1])
    return encoder, scaler, model, X, feature_names


class TestOnnxParity(unittest.TestCase):
    """A6: o grafo exportado tem que ser o mesmo modelo."""

    @classmethod
    def setUpClass(cls) -> None:
        _, scaler, model, cls.X, _ = _fit_all(_dataset())
        cls.exportable = build_exportable(model, scaler)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls._tmp.name) / "model.onnx"
        export_onnx(cls.exportable, cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _compare(self, X: np.ndarray) -> None:
        obtido = run_onnx(self.model_path, X)
        with torch.inference_mode():
            esperado = self.exportable(
                torch.tensor(X, dtype=torch.float32)
            ).numpy()
        np.testing.assert_allclose(obtido, esperado, atol=PARITY_TOLERANCE)

    def test_parity_on_training_data(self) -> None:
        self._compare(self.X)

    def test_parity_on_single_record(self) -> None:
        self._compare(self.X[:1])

    def test_parity_on_random_input(self) -> None:
        rng = np.random.default_rng(7)
        self._compare(rng.normal(size=(200, N_FEATURES)) * 1000)

    def test_batch_axis_is_dynamic(self) -> None:
        for n in (1, 3, 97):
            with self.subTest(batch=n):
                self.assertEqual(run_onnx(self.model_path, self.X[:n]).shape, (n,))

    def test_output_is_a_probability(self) -> None:
        saida = run_onnx(self.model_path, self.X)
        self.assertTrue(((saida >= 0.0) & (saida <= 1.0)).all())

    def test_scaler_is_inside_the_graph(self) -> None:
        """A entrada do ONNX é o vetor cru: normalizar de novo mudaria a saída."""
        crua = run_onnx(self.model_path, self.X)
        renormalizada = run_onnx(self.model_path, np.zeros_like(self.X))
        self.assertFalse(np.allclose(crua, renormalizada))


class TestBundleLayout(unittest.TestCase):
    """A2: o bundle é autocontido e íntegro."""

    @classmethod
    def setUpClass(cls) -> None:
        dataset = _dataset()
        encoder, scaler, model, _, feature_names = _fit_all(dataset)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name) / "bundle"
        csv = Path(cls._tmp.name) / "loan_data.csv"
        dataset.to_csv(csv, index=False)
        export_bundle(
            output_dir=cls.dir, model=model, scaler=scaler, encoder=encoder,
            records=dataset, feature_names=feature_names,
            metrics={"AUROC": 0.5}, dataset_path=csv,
            epochs=1, learning_rate=0.01,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_expected_files(self) -> None:
        self.assertEqual(
            sorted(p.name for p in self.dir.iterdir()),
            ["SHA256SUMS", "golden.json", "model.onnx",
             "model_card.json", "preprocessing.json"],
        )

    def test_onnx_is_self_contained(self) -> None:
        """Sem model.onnx.data ao lado: o artefato viaja sozinho."""
        self.assertFalse((self.dir / "model.onnx.data").exists())

    def test_checksums_match(self) -> None:
        for linha in (self.dir / "SHA256SUMS").read_text().strip().splitlines():
            esperado, nome = linha.split("  ")
            digest = hashlib.sha256((self.dir / nome).read_bytes()).hexdigest()
            self.assertEqual(digest, esperado, f"hash divergente em {nome}")

    def test_preprocessing_declares_full_contract(self) -> None:
        spec = json.loads((self.dir / "preprocessing.json").read_text())
        self.assertEqual(spec["n_features"], N_FEATURES)
        self.assertEqual(len(spec["feature_names"]), N_FEATURES)
        self.assertEqual(spec["unknown_category_policy"], "reject")
        self.assertIn("person_gender", spec["dropped_columns"])

    def test_golden_payload_excludes_target_and_gender(self) -> None:
        golden = json.loads((self.dir / "golden.json").read_text())
        for record in golden["records"]:
            self.assertNotIn("loan_status", record["payload"])
            self.assertNotIn("person_gender", record["payload"])

    def test_golden_covers_every_category(self) -> None:
        """Amostragem pura perderia 'OTHER' (0,26% do dataset real)."""
        golden = json.loads((self.dir / "golden.json").read_text())
        for column, categories in ONE_HOT_CATEGORIES.items():
            vistas = {r["payload"][column] for r in golden["records"]}
            self.assertEqual(set(categories) - vistas, set(), f"faltou em {column}")


@unittest.skipUnless(BUNDLE_DIR.is_dir(), "bundle ainda não exportado")
class TestPublishedBundle(unittest.TestCase):
    """Valida o bundle realmente publicado em data/models/bundle."""

    def test_golden_reproduces(self) -> None:
        golden = json.loads((BUNDLE_DIR / "golden.json").read_text())
        dataset = pd.DataFrame([r["payload"] for r in golden["records"]])
        esperado = np.array([r["default_probability"] for r in golden["records"]])

        encoder, _, _, _ = fit_encoder(
            pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "loan_data.csv")
        )
        obtido = run_onnx(BUNDLE_DIR / "model.onnx", transform_records(encoder, dataset))

        np.testing.assert_allclose(obtido, esperado, atol=golden["tolerance"])

    def test_checksums_match(self) -> None:
        for linha in (BUNDLE_DIR / "SHA256SUMS").read_text().strip().splitlines():
            esperado, nome = linha.split("  ")
            digest = hashlib.sha256((BUNDLE_DIR / nome).read_bytes()).hexdigest()
            self.assertEqual(digest, esperado, f"hash divergente em {nome}")


if __name__ == "__main__":
    unittest.main()
