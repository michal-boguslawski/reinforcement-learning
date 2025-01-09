from random import sample, choices
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, size=1000):
        self.buffer = []
        self.weights = []
        self.size = size

    def reset(self):
        self.buffer = []

    def append(self, element, weight):
        self.buffer.append(element)
        self.weights.append(weight.item())
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
            self.weights.pop(0)

    def sample(self, n, temperature=1):
        max_weight = 100
        weights = np.clip(np.array(self.weights), 0, max_weight)
        weights = np.power(weights, 1 / temperature)
        weights = np.maximum(weights, 0) + 1e-9
        # Normalize weights
        weights_sum = np.sum(weights)
        if not np.isfinite(weights_sum) or weights_sum == 0:
            weights = np.ones_like(weights) / len(weights)  # Fallback: uniform weights
        else:
            weights = weights / weights_sum
        samples = choices(self.buffer, weights=weights, k=n)
        columns = list(zip(*samples))
        elements = [torch.cat(column, dim=0) for column in columns]
        return elements
