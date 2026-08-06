"""Testes das funções de treinamento do módulo training."""

import copy
import unittest
import random

import numpy as np
import torch

from src.training.train import accuracy_fn, train_nn
from src.models.NeuralNetworkV0 import NeuralNetworkV0
from src.utils.config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fixa as seeds de todos os geradores de números aleatórios para
    garantir reprodutibilidade do treinamento (CPU e GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TestTraining(unittest.TestCase):
    def test_training_is_deterministic(self) -> None:
        """Duas execuções do treinamento com a mesma seed devem produzir
        pesos idênticos no modelo."""
        X = torch.randn(100, 23)
        y = torch.randint(0, 2, (100,)).float()

        set_seed(RANDOM_SEED)
        model1 = NeuralNetworkV0()
        train_nn(10, model1, 0.01, X, y, X, y)

        set_seed(RANDOM_SEED)
        model2 = NeuralNetworkV0()
        train_nn(10, model2, 0.01, X, y, X, y)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2),
                            "Os pesos das duas execuções não são idênticos.")

    def test_test_data_does_not_affect_training(self) -> None:
        """Os dados de teste não devem influenciar o treinamento:
        dois modelos treinados com o mesmo X_train/y_train, mas X_test/y_test
        diferentes, devem ter pesos idênticos."""
        X_train = torch.randn(80, 23)
        y_train = torch.randint(0, 2, (80,)).float()
        X_test1 = torch.randn(20, 23)
        y_test1 = torch.randint(0, 2, (20,)).float()
        X_test2 = torch.randn(50, 23)
        y_test2 = torch.randint(0, 2, (50,)).float()

        set_seed(RANDOM_SEED)
        model1 = NeuralNetworkV0()
        train_nn(10, model1, 0.01, X_train, y_train, X_test1, y_test1)

        set_seed(RANDOM_SEED)
        model2 = NeuralNetworkV0()
        train_nn(10, model2, 0.01, X_train, y_train, X_test2, y_test2)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2),
                            "Os pesos diferem — os dados de teste estão "
                            "influenciando o treinamento.")

    def test_parameters_change_after_training(self) -> None:
        """Após o treinamento, os parâmetros do modelo devem ser diferentes
        do estado inicial, confirmando que o otimizador de fato atualizou
        os pesos."""
        X = torch.randn(50, 23)
        y = torch.randint(0, 2, (50,)).float()

        set_seed(RANDOM_SEED)
        model = NeuralNetworkV0()
        params_before = copy.deepcopy(list(model.parameters()))

        train_nn(5, model, 0.01, X, y, X, y)

        any_changed = False
        for before, after in zip(params_before, model.parameters()):
            if not torch.equal(before, after):
                any_changed = True
                break
        self.assertTrue(any_changed,
                        "Nenhum parâmetro do modelo foi alterado após o "
                        "treinamento.")

class TestAccuracyFn(unittest.TestCase):
    def test_all_correct(self) -> None:
        """Quando todas as predições batem com os rótulos, a acurácia deve ser
        100%. Garante que o cálculo percentual não introduz erro de
        arredondamento."""
        y_true = torch.tensor([1, 0, 1, 0, 1])
        y_pred = torch.tensor([1, 0, 1, 0, 1])
        result = accuracy_fn(y_true, y_pred)
        self.assertAlmostEqual(result, 100.0)

    def test_none_correct(self) -> None:
        """Quando nenhuma predição acerta, a acurácia deve ser 0%.
        Garante que a função não retorna valores negativos nem NaN."""
        y_true = torch.tensor([1, 1, 1, 1, 1])
        y_pred = torch.tensor([0, 0, 0, 0, 0])
        result = accuracy_fn(y_true, y_pred)
        self.assertAlmostEqual(result, 0.0)

    def test_half_correct(self) -> None:
        """Com 2 acertos em 4 amostras, a acurácia deve ser exatamente 50%.
        Valida que a divisão e multiplicação por 100 estão corretas."""
        y_true = torch.tensor([1, 0, 1, 0])
        y_pred = torch.tensor([1, 1, 0, 0])
        result = accuracy_fn(y_true, y_pred)
        self.assertAlmostEqual(result, 50.0)

    def test_with_floats(self) -> None:
        """Tensores float32 devem funcionar da mesma forma que inteiros.
        A função usa torch.eq, que compara elementos independentemente do
        dtype — 2 acertos em 3 amostras = 66.666...%."""
        y_true = torch.tensor([1.0, 0.0, 1.0])
        y_pred = torch.tensor([1.0, 0.0, 2.0])
        result = accuracy_fn(y_true, y_pred)
        self.assertAlmostEqual(result, 200.0 / 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
