#!/usr/bin/env python3
"""
Triple-Stream Classifier for Handshape Recognition

This classifier combines three complementary sources of information:
1. Sign embeddings from SignGNN (temporal dynamics)
2. Handshape embeddings from HandshapeGNN (static configurations)
3. Raw landmark coordinates (direct geometric features)

Reference: "Improving Handshape Representations for Sign Language Processing:
           A Graph Neural Network Approach" (EMNLP 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and dropout.

    Args:
        in_features (int): Input feature dimension
    """

    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class TripleStreamClassifier(nn.Module):
    """
    Triple-stream architecture that combines sign embeddings, handshape embeddings,
    and raw landmarks for handshape classification.

    Architecture:
        - Sign branch: processes 32-dim sign embeddings
        - Handshape branch: processes 32-dim handshape embeddings
        - Landmark branch: processes 63-dim raw coordinates (21 landmarks × 3)
        - Each branch: Linear → BatchNorm → ReLU → Dropout → ResidualBlock
        - Combined features are concatenated (192-dim total)
        - Final classifier: Linear → BatchNorm → ReLU → Dropout → ResidualBlock → Linear

    Args:
        num_classes (int): Number of handshape classes
        sign_dim (int): Dimension of sign embeddings (default: 32)
        handshape_dim (int): Dimension of handshape embeddings (default: 32)
        landmark_dim (int): Dimension of raw landmarks (default: 63)
        hidden_dim (int): Dimension of hidden layers (default: 64)
    """

    def __init__(self, num_classes, sign_dim=32, handshape_dim=32,
                 landmark_dim=63, hidden_dim=64):
        super().__init__()

        # Sign embeddings branch
        self.sign_branch = nn.Sequential(
            nn.Linear(sign_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_dim)
        )

        # Handshape embeddings branch
        self.handshape_branch = nn.Sequential(
            nn.Linear(handshape_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_dim)
        )

        # Raw landmarks branch
        self.landmark_branch = nn.Sequential(
            nn.Linear(landmark_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_dim)
        )

        # Combined features dimension: 64 * 3 = 192
        combined_dim = hidden_dim * 3

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, sign_emb, handshape_emb, landmarks):
        """
        Forward pass through the triple-stream architecture.

        Args:
            sign_emb (torch.Tensor): Sign embeddings [batch_size, sign_dim]
            handshape_emb (torch.Tensor): Handshape embeddings [batch_size, handshape_dim]
            landmarks (torch.Tensor): Raw landmarks [batch_size, landmark_dim]

        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        # Process each branch
        sign_features = self.sign_branch(sign_emb)
        handshape_features = self.handshape_branch(handshape_emb)
        landmark_features = self.landmark_branch(landmarks)

        # Combine all features
        combined = torch.cat([sign_features, handshape_features, landmark_features], dim=1)

        # Final classification
        return self.classifier(combined)

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
