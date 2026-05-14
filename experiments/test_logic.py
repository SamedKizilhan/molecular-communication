import torch
L = 10
all_pos = torch.randn(5, 3) * 15
mask_high = all_pos > L
all_pos[mask_high] = 2 * L - all_pos[mask_high]
print("SUCCESS!")