import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionClassifier(nn.Module):
    """
    Attention-based neural network for gene expression classification.
    """

    def __init__(self, input_dim=2000, hidden_dim=128, num_classes=2):
        super(AttentionClassifier, self).__init__()

        # Feature transformation layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attention mechanism: computes importance weights for features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        features = self.feature_extractor(x)

        # Apply attention to focus on important features
        # Note: In this simple structure, we use attention to weigh feature importance
        weights = self.attention(features)
        context = features * weights

        logits = self.classifier(context)
        return logits, weights