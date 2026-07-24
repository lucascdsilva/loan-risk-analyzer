import numpy as np
import torch
from torch import nn
from src.utils.config import RANDOM_SEED

def to_tensor(
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        device: str):
    """Converte arrays NumPy para tensores PyTorch float32 e os
    transfere para o device especificado (CPU/GPU)."""
    # Convertendo arrays numpy em tensores
    X_train_tensor = torch.from_numpy(X_train).type(torch.float)
    y_train_tensor = torch.from_numpy(y_train).type(torch.float)
    X_test_tensor = torch.from_numpy(X_test).type(torch.float)
    y_test_tensor = torch.from_numpy(y_test).type(torch.float)

    # Transfere os dados para a CPU ou GPU
    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def train_nn(epochs: int, model: nn.Module, learning_rate: float, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor) -> None:
    """Loop de treinamento com otimizador Adam e BCEWithLogitsLoss.
    A cada época: forward pass, cálculo de loss/acurácia, backpropagation.
    Em modo de inferência, avalia no conjunto de teste e imprime métricas
    a cada 500 épocas."""
    test_pred = None
    torch.manual_seed(RANDOM_SEED)

    # Cria o otimizador para os parâmetros da rede
    #optimizer = torch.optim.SGD(params=nn_model.parameters(), lr=0.01) # taxa de aprendizado 0.01
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

    print("\n--- Treinamento ---")
    for epoch in range(epochs):
        model.train()
        # 1. Forward pass (model outputs raw logits)
        y_logits = model(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
        y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels
        
        # 2. Calculate loss/accuracy
        # Passa direto y_logits, porque está usando a função BCEWithLogitsLoss(). 
        # Se usar BCELoss(), precisa aplicar a sigmoide antes (torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train) 
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred) 
        
         # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backwards
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        ### Testing
        model.eval()
        
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test).squeeze() 
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. Caculate loss/accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calcula a acurácia (em %) comparando predições com rótulos reais
    elemento a elemento."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc