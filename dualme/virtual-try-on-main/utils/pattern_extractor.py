import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_candidate_blocks(model, layer_types=(nn.Conv2d, nn.Linear), min_size=64):
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            weight = getattr(module, 'weight', None)
            if weight is not None and weight.numel() >= min_size:
                candidates.append((name, module, weight.clone().detach()))
    return candidates

def compute_block_similarity(block_a, block_b, method='cosine'):
    if method == 'cosine':
        a = block_a.flatten(start_dim=1)
        b = block_b.flatten(start_dim=1)
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        similarity = torch.mm(a, b.T).mean().item()
    elif method == 'euclidean':
        diff = block_a - block_b
        similarity = -torch.norm(diff).item()
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    return similarity

def find_similar_block_pairs(candidates, threshold=0.95):
    similar_pairs = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            name_a, mod_a, w_a = candidates[i]
            name_b, mod_b, w_b = candidates[j]
            if w_a.shape == w_b.shape:
                sim = compute_block_similarity(w_a, w_b)
                if sim >= threshold:
                    similar_pairs.append(((name_a, mod_a), (name_b, mod_b), sim))
    return similar_pairs
