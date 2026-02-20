#!/usr/bin/env python3
"""
Baseline MLP Classifier for Handshape Recognition

A simple multi-layer perceptron that operates directly on raw landmark coordinates
without leveraging graph structure or pre-trained embeddings.

Reference: "Improving Handshape Representations for Sign Language Processing:
           A Graph Neural Network Approach" (EMNLP 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMLP(nn.Module):
    """
    Baseline feedforward neural network for handshape classification.

    Architecture:
        - Input: 63-dim raw landmarks (21 landmarks × 3 coordinates)
        - Three linear layers with batch normalization and dropout
        - Output: class logits for handshape prediction

    Args:
        num_classes (int): Number of handshape classes
        input_dim (int): Dimension of input features (default: 63)
        hidden_dim (int): Dimension of hidden layers (default: 256)
        dropout (float): Dropout probability (default: 0.3)
    """

    def __init__(self, num_classes, input_dim=63, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input landmarks [batch_size, input_dim]

        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        return self.network(x)

    def save_checkpoint(self, path, epoch, optimizer, scheduler, val_accuracy,
                       val_f1, class_metrics, config):
        """
        Save model checkpoint with training state.

        Args:
            path (str): Path to save checkpoint
            epoch (int): Current epoch number
            optimizer: Optimizer state
            scheduler: Scheduler state
            val_accuracy (float): Validation accuracy
            val_f1 (float): Validation macro F1 score
            class_metrics (dict): Per-class metrics
            config (dict): Training configuration
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'class_metrics': class_metrics,
            'config': config
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_from_checkpoint(cls, path, num_classes, device='cpu'):
        """
        Load model from checkpoint.

        Args:
            path (str): Path to checkpoint file
            num_classes (int): Number of handshape classes
            device (str): Device to load model on

        Returns:
            tuple: (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device)

        model = cls(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint
