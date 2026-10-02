import torch

def euclidean_distance(feature1, feature2):
    return torch.norm(feature1 - feature2)


def chebyshev_distance(feature1, feature2):
    return torch.max(torch.abs(feature1 - feature2))

def cosine_distance(feature1, feature2):
    cosine_similarity = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0)
    return 1 - cosine_similarity
