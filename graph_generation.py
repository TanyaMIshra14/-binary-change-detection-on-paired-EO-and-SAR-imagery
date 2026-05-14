import torch

def get_grid_size(image_size: int, stride: int = 32) -> int:
    return image_size // stride

def create_grid_graph(H, W):
    edges = []
    for y in range(H):
        for x in range(W):
            idx = y * W + x
            # Right neighbor
            if x < W - 1:
                right = y * W + (x + 1)
                edges.append([idx, right])
                edges.append([right, idx])
            # Bottom neighbor
            if y < H - 1:
                bottom = (y + 1) * W + x
                edges.append([idx, bottom])
                edges.append([bottom, idx])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    return edge_index