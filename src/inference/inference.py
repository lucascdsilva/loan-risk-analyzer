import torch
from torch import nn

def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Executa inferência: aplica sigmoid aos logits da rede, arredonda para
    obter predições binárias (0/1) e retorna como int32."""
    model.eval()
    with torch.inference_mode():
        logits = model(X).squeeze() 
        y_pred = torch.round(torch.sigmoid(logits))
    
    return y_pred.to(torch.int32)