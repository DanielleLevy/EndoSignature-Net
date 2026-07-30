import torch
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    """
    Handles the training process of the AttentionClassifier.
    """

    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x, y):
        """
        Performs a single training step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits, weights = self.model(x)

        # Calculate loss
        loss = self.criterion(logits, y)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()