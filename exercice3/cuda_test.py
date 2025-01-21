import torch

# Vérifiez si un GPU AMD est détecté
if torch.cuda.is_available():
    print("GPU AMD détecté avec PyTorch!")
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
else:
    print("Aucun GPU AMD détecté.")
