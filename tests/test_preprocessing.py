"""Testes da escala ordinal de escolaridade aplicada por ``prepare_frame`` """

import unittest

import pandas as pd

from src.preprocessing.transform import EDUCATION_ORDER, prepare_frame


def _frame(education: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "person_age": 30.0,
        "person_gender": "female",
        "person_education": education,
        "person_income": 60000.0,
        "person_emp_exp": 5,
        "person_home_ownership": "RENT",
        "loan_amnt": 10000.0,
        "loan_intent": "PERSONAL",
        "loan_int_rate": 12.5,
        "loan_percent_income": 0.17,
        "cb_person_cred_hist_length": 5.0,
        "credit_score": 650,
        "previous_loan_defaults_on_file": "No",
        "loan_status": 0,
    }])


def _education_level(education: str) -> int:
    return prepare_frame(_frame(education))["person_education"].iloc[0]


class TestEducationOrdinal(unittest.TestCase):
    def test_education_ordinal_bachelor(self) -> None:
        self.assertEqual(_education_level("Bachelor"), 2)

    def test_education_ordinal_master_greater_than_bachelor(self) -> None:
        self.assertGreater(_education_level("Master"), _education_level("Bachelor"))

    def test_scale_is_monotonic(self) -> None:
        """A ordem dos níveis é o que dá sentido ao tratamento ordinal."""
        niveis = [_education_level(e) for e in EDUCATION_ORDER]
        self.assertEqual(niveis, sorted(niveis))


if __name__ == "__main__":
    unittest.main()
