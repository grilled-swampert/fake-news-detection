import torch
print(torch.cuda.is_available())
# This should print True
print(torch.cuda.get_device_name(0))
# This should print NVIDIA RTX A5000