import torch

print(torch.__version__)

if torch.cuda.is_available():
    print("cuda is available")
else:
    print("We find no GPU, CPU mode is on.")
