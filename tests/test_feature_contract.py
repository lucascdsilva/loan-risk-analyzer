"""Testes do contrato de features (seção 3 de docs/PLANO_ARQUITETURA.md).

Cobrem o que o pipeline de inferência precisa garantir: mesmas colunas na
mesma ordem para lote e para registro único, rejeição de categoria
desconhecida e ausência de efeito colateral no DataFrame de entrada.
"""

import unittest

import numpy as np
import pandas as pd

from src.preprocessing.transform import (
    CATEGORICAL_COLUMNS,
    DROPPED_COLUMNS,
    N_FEATURES,
    ONE_HOT_CATEGORIES,
    TARGET_COLUMN,
    fit_encoder,
    fit_scaler,
    prepare_frame,
    transform_records,
)

EXPECTED_FEATURE_ORDER = [
    "person_home_ownership_MORTGAGE", "person_home_ownership_OTHER",
    "person_home_ownership_OWN", "person_home_ownership_RENT",
    "loan_intent_DEBTCONSOLIDATION", "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL",
    "loan_intent_PERSONAL", "loan_intent_VENTURE",
    "previous_loan_defaults_on_file_No", "previous_loan_defaults_on_file_Yes",
    "person_age", "person_education", "person_income", "person_emp_exp",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score",
]


def _row(**kwargs) -> dict:
    defaults = dict(
        person_age=30.0,
        person_gender="female",
        person_education="Bachelor",
        person_income=60000.0,
        person_emp_exp=5,
        person_home_ownership="RENT",
        loan_amnt=10000.0,
        loan_intent="PERSONAL",
        loan_int_rate=12.5,
        loan_percent_income=0.17,
        cb_person_cred_hist_length=5.0,
        credit_score=650,
        previous_loan_defaults_on_file="No",
        loan_status=0,
    )
    defaults.update(kwargs)
    return defaults


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [_row()])


def _training_frame() -> pd.DataFrame:
    """Amostra cobrindo todas as categorias, para ajustar encoder e scaler."""
    rows = []
    for i, home in enumerate(ONE_HOT_CATEGORIES["person_home_ownership"]):
        for j, intent in enumerate(ONE_HOT_CATEGORIES["loan_intent"]):
            rows.append(_row(
                person_home_ownership=home,
                loan_intent=intent,
                previous_loan_defaults_on_file="Yes" if (i + j) % 2 else "No",
                person_income=50000.0 + 1000 * (i + j),
                loan_status=(i + j) % 2,
            ))
    return pd.DataFrame(rows)


class TestFeatureContract(unittest.TestCase):
    def test_feature_count_is_21(self) -> None:
        self.assertEqual(N_FEATURES, 21)

    def test_feature_names_and_order(self) -> None:
        _, _, _, names = fit_encoder(_training_frame())
        self.assertEqual(names, EXPECTED_FEATURE_ORDER)

    def test_matrix_width_matches_contract(self) -> None:
        _, X, _, _ = fit_encoder(_training_frame())
        self.assertEqual(X.shape[1], N_FEATURES)

    def test_gender_is_dropped(self) -> None:
        """Decisão D2: gênero não é variável de decisão de crédito."""
        self.assertIn("person_gender", DROPPED_COLUMNS)
        _, _, _, names = fit_encoder(_training_frame())
        self.assertEqual([n for n in names if "gender" in n], [])

    def test_gender_is_not_a_categorical_column(self) -> None:
        self.assertNotIn("person_gender", CATEGORICAL_COLUMNS)

    def test_target_is_not_a_feature(self) -> None:
        _, _, _, names = fit_encoder(_training_frame())
        self.assertNotIn(TARGET_COLUMN, names)


class TestSingleRecordParity(unittest.TestCase):
    """O caso que quebrava antes: inferência sobre um registro só."""

    def test_single_record_has_full_width(self) -> None:
        encoder, _, _, _ = fit_encoder(_training_frame())
        X = transform_records(encoder, _frame(_row(person_home_ownership="RENT")))
        self.assertEqual(X.shape, (1, N_FEATURES))

    def test_single_record_matches_batch(self) -> None:
        encoder, _, _, _ = fit_encoder(_training_frame())
        batch = _frame(
            _row(person_home_ownership="OWN", loan_intent="MEDICAL"),
            _row(person_home_ownership="MORTGAGE", loan_intent="VENTURE"),
        )
        from_batch = transform_records(encoder, batch)
        one_by_one = np.vstack([
            transform_records(encoder, batch.iloc[[i]]) for i in range(len(batch))
        ])
        np.testing.assert_array_equal(from_batch, one_by_one)

    def test_works_without_target_column(self) -> None:
        encoder, _, _, _ = fit_encoder(_training_frame())
        sem_alvo = _frame().drop(columns=[TARGET_COLUMN])
        self.assertEqual(transform_records(encoder, sem_alvo).shape[1], N_FEATURES)


class TestUnknownCategories(unittest.TestCase):
    """RF-ML-04: rejeitar, nunca zerar em silêncio."""

    def test_unknown_one_hot_category_raises(self) -> None:
        encoder, _, _, _ = fit_encoder(_training_frame())
        with self.assertRaises(ValueError):
            transform_records(encoder, _frame(_row(loan_intent="CRIPTOMOEDA")))

    def test_unknown_education_raises(self) -> None:
        with self.assertRaises(ValueError):
            prepare_frame(_frame(_row(person_education="Pós-doutorado")))


class TestNoSideEffects(unittest.TestCase):
    def test_input_frame_is_not_mutated(self) -> None:
        original = _frame()
        antes = original.copy(deep=True)
        fit_encoder(original)
        pd.testing.assert_frame_equal(original, antes)

    def test_prepare_frame_does_not_mutate(self) -> None:
        original = _frame()
        antes = original.copy(deep=True)
        prepare_frame(original)
        pd.testing.assert_frame_equal(original, antes)


class TestScaler(unittest.TestCase):
    def test_scaler_fitted_on_train_only(self) -> None:
        _, X, _, _ = fit_encoder(_training_frame())
        scaler = fit_scaler(X)
        self.assertEqual(scaler.mean_.shape, (N_FEATURES,))

    def test_scaler_is_reusable_on_single_record(self) -> None:
        encoder, X, _, _ = fit_encoder(_training_frame())
        scaler = fit_scaler(X)
        um = transform_records(encoder, _frame())
        self.assertEqual(scaler.transform(um).shape, (1, N_FEATURES))


if __name__ == "__main__":
    unittest.main()
