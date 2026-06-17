"""Testes do pré-processamento e split do dataset de empréstimos."""

import unittest

from src.data.loan_loader import LoanRecord
from src.preprocessing.transform import (
    CleanedRecord,
    clean_dataset,
    encode_record,
    split_data,
)


def _record(**kwargs) -> LoanRecord:
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
    return LoanRecord(**defaults)


class TestEncoding(unittest.TestCase):
    def test_gender_female_encoded_as_1(self) -> None:
        result = encode_record(_record(person_gender="female"))
        self.assertEqual(result.gender_female, 1)

    def test_gender_male_encoded_as_0(self) -> None:
        result = encode_record(_record(person_gender="male"))
        self.assertEqual(result.gender_female, 0)

    def test_education_ordinal_bachelor(self) -> None:
        result = encode_record(_record(person_education="Bachelor"))
        self.assertEqual(result.education_level, 2)

    def test_education_ordinal_master_greater_than_bachelor(self) -> None:
        bachelor = encode_record(_record(person_education="Bachelor"))
        master = encode_record(_record(person_education="Master"))
        self.assertGreater(master.education_level, bachelor.education_level)

    def test_previous_default_yes_encoded_as_1(self) -> None:
        result = encode_record(_record(previous_loan_defaults_on_file="Yes"))
        self.assertEqual(result.previous_default, 1)

    def test_previous_default_no_encoded_as_0(self) -> None:
        result = encode_record(_record(previous_loan_defaults_on_file="No"))
        self.assertEqual(result.previous_default, 0)

    def test_result_is_cleaned_record(self) -> None:
        result = encode_record(_record())
        self.assertIsInstance(result, CleanedRecord)


class TestCleanDataset(unittest.TestCase):
    def test_clean_dataset_applies_to_all(self) -> None:
        records = [_record() for _ in range(5)]
        cleaned = clean_dataset(records)
        self.assertEqual(len(cleaned), 5)


class TestSplitData(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        records = clean_dataset([_record(loan_status=i % 2) for i in range(10)])
        train1, test1 = split_data(records, test_ratio=0.2, seed=42)
        train2, test2 = split_data(records, test_ratio=0.2, seed=42)
        self.assertEqual(
            [r.loan_status for r in test1],
            [r.loan_status for r in test2],
        )

    def test_split_proportions(self) -> None:
        records = clean_dataset([_record() for _ in range(10)])
        train, test = split_data(records, test_ratio=0.2)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(train), 8)

    def test_split_invalid_ratio(self) -> None:
        records = clean_dataset([_record()])
        with self.assertRaises(ValueError):
            split_data(records, test_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
