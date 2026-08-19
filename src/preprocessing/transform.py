"""Pré-processamento do dataset de risco de crédito.

Cobre codificação de variáveis categóricas, vetorização com NumPy (matriz de
features pronta para o modelo) e split treino/teste. A conversão para tensores
entra nas etapas seguintes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from src.data.loan_loader import LoanRecord
from src.utils.config import RANDOM_SEED

# Escolaridade tratada como ordinal (quanto maior, mais escolaridade).
EDUCATION_ORDER = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4,
}

# ---------------------------------------------------------------------------
# Contrato de features (v0.4.0) — 21 colunas
# ---------------------------------------------------------------------------

TARGET_COLUMN = "loan_status"

# Descartado: gênero não é usado como variável de decisão de crédito.
DROPPED_COLUMNS: Tuple[str, ...] = ("person_gender",)

# Categorias fixadas explicitamente. Garante as mesmas 21 colunas no
# treino e na inferência de um único registro
ONE_HOT_CATEGORIES: Dict[str, List[str]] = {
    "person_home_ownership": ["MORTGAGE", "OTHER", "OWN", "RENT"],
    "loan_intent": [
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
        "MEDICAL", "PERSONAL", "VENTURE",
    ],
    "previous_loan_defaults_on_file": ["No", "Yes"],
}

CATEGORICAL_COLUMNS = list(ONE_HOT_CATEGORIES)

N_FEATURES = sum(len(c) for c in ONE_HOT_CATEGORIES.values()) + 9


def build_encoder() -> ColumnTransformer:
    """Cria o codificador de features (ainda não ajustado).

    One-hot nas categóricas com categorias fixas, ``passthrough`` no resto.
    """
    return ColumnTransformer(
        transformers=[(
            "one_hot",
            OneHotEncoder(
                categories=[ONE_HOT_CATEGORIES[c] for c in CATEGORICAL_COLUMNS],
                sparse_output=False,
                handle_unknown="error",
            ),
            CATEGORICAL_COLUMNS,
        )],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def prepare_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """Descarta colunas fora do contrato e aplica a escala ordinal.
    Raises:
        ValueError: se aparecer uma escolaridade fora de ``EDUCATION_ORDER``.
    """
    prepared = dataset.drop(
        columns=[c for c in DROPPED_COLUMNS if c in dataset.columns]
    )

    education = prepared["person_education"].map(EDUCATION_ORDER)
    if education.isna().any():
        desconhecidas = sorted(
            set(prepared.loc[education.isna(), "person_education"])
        )
        raise ValueError(f"Escolaridade desconhecida: {desconhecidas}")
    prepared["person_education"] = education

    return prepared


def fit_encoder(
    dataset: pd.DataFrame,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray, List[str]]:
    """Ajusta o codificador no dataset de treino.

    Args:
        dataset: DataFrame com as colunas do CSV, incluindo ``loan_status``.

    Returns:
        Tupla ``(encoder, X, y, feature_names)``. O ``encoder`` retornado é o
        artefato que precisa ser persistido para que a inferência reproduza
        exatamente esta transformação.
    """
    prepared = prepare_frame(dataset)
    target = prepared[TARGET_COLUMN].to_numpy(dtype=np.int64)
    features_df = prepared.drop(columns=[TARGET_COLUMN])

    encoder = build_encoder()
    features = encoder.fit_transform(features_df).astype(np.float64)

    return encoder, features, target, list(encoder.get_feature_names_out())


def transform_records(
    encoder: ColumnTransformer, dataset: pd.DataFrame
) -> np.ndarray:
    """Aplica um codificador já ajustado a registros novos.

    Diferente de :func:`fit_encoder`, não exige ``loan_status`` — é o caminho
    usado na inferência, onde o alvo é justamente o que se quer estimar.
    """
    prepared = prepare_frame(dataset)
    features_df = prepared.drop(columns=[TARGET_COLUMN], errors="ignore")
    return encoder.transform(features_df).astype(np.float64)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Ajusta a normalização **somente** no treino, para evitar vazamento. """
    return StandardScaler().fit(X_train)


@dataclass(frozen=True)
class CleanedRecord:
    """Registro com features numéricas e categóricas já codificadas."""

    person_age: float
    person_income: float
    person_emp_exp: int
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    gender_female: int      # 1 = feminino, 0 = masculino
    education_level: int    # nível ordinal de escolaridade
    home_ownership: str     # categoria preservada (one-hot na vetorização)
    loan_intent: str        # finalidade do empréstimo
    previous_default: int   # 1 = "Yes", 0 = "No"
    loan_status: int        # variável alvo: 0 = sem default, 1 = default


def encode_record(record: LoanRecord) -> CleanedRecord:
    """Codifica as variáveis categóricas de um único LoanRecord."""
    return CleanedRecord(
        person_age=record.person_age,
        person_income=record.person_income,
        person_emp_exp=record.person_emp_exp,
        loan_amnt=record.loan_amnt,
        loan_int_rate=record.loan_int_rate,
        loan_percent_income=record.loan_percent_income,
        cb_person_cred_hist_length=record.cb_person_cred_hist_length,
        credit_score=record.credit_score,
        gender_female=1 if record.person_gender.lower() == "female" else 0,
        education_level=EDUCATION_ORDER.get(record.person_education, -1),
        home_ownership=record.person_home_ownership,
        loan_intent=record.loan_intent,
        previous_default=1 if record.previous_loan_defaults_on_file == "Yes" else 0,
        loan_status=record.loan_status,
    )

def smote_oversampling(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    smote = SMOTE(random_state=RANDOM_SEED)
    return smote.fit_resample(X, y)

def clean_dataset(records: Sequence[LoanRecord]) -> List[CleanedRecord]:
    """Codifica todos os registros, descartando linhas com erro de parsing."""
    result: List[CleanedRecord] = []
    for r in records:
        try:
            result.append(encode_record(r))
        except (ValueError, KeyError):
            continue
    return result
# Features numéricas e ordinais já prontas para entrar direto na matriz.
NUMERIC_FEATURES = (
    "person_age", "person_income", "person_emp_exp", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "gender_female", "education_level", "previous_default",
)

# Categóricas restantes que recebem codificação one-hot na vetorização.
ONEHOT_FEATURES = ("home_ownership", "loan_intent")


def build_feature_matrix(
    records: Sequence[CleanedRecord],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Vetoriza os registros codificados em uma matriz de features NumPy.

    As features numéricas/ordinais são empilhadas diretamente e as categóricas
    restantes (``home_ownership``, ``loan_intent``) recebem codificação one-hot.

    Args:
        records: registros já passados por :func:`encode_record`.

    Returns:
        Tupla ``(X, y, feature_names)`` onde ``X`` é a matriz
        ``(n_amostras, n_features)`` em ``float64``, ``y`` é o vetor alvo
        ``loan_status`` em ``int64`` e ``feature_names`` são os nomes das
        colunas de ``X`` na ordem.
    """
    items = list(records)
    if not items:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            [],
        )

    numeric = np.array(
        [[getattr(r, name) for name in NUMERIC_FEATURES] for r in items],
        dtype=np.float64,
    )

    feature_names = list(NUMERIC_FEATURES)
    onehot_blocks: List[np.ndarray] = []
    for field in ONEHOT_FEATURES:
        values = np.array([getattr(r, field) for r in items])
        categories = np.unique(values)
        # Broadcasting: (n, 1) == (1, k) -> matriz one-hot (n, k).
        onehot_blocks.append((values[:, None] == categories[None, :]).astype(np.float64))
        feature_names.extend(f"{field}={c}" for c in categories)

    features = np.hstack([numeric, *onehot_blocks])
    target = np.array([r.loan_status for r in items], dtype=np.int64)
    return features, target, feature_names


def split_data(
    records: Sequence[CleanedRecord],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[CleanedRecord], List[CleanedRecord]]:
    """Divide os registros em treino e teste de forma determinística.

    Args:
        records: registros já pré-processados.
        test_ratio: fração reservada para teste (0 < ratio < 1).
        seed: semente para embaralhamento reprodutível.

    Returns:
        Tupla (treino, teste).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio deve estar entre 0 e 1.")
    items = list(records)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    shuffled = [items[i] for i in order]
    cut = int(len(items) * (1.0 - test_ratio))
    return shuffled[:cut], shuffled[cut:]
