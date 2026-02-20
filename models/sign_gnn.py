#!/usr/bin/env python3
"""
SignGNN: Temporal Graph Neural Network for Sign Language Processing

This model processes full signing sequences with both spatial (anatomical hand connections)
and temporal (across-frame) edges to learn representations of temporal dynamics.

Reference: "Improving Handshape Representations for Sign Language Processing:
           A Graph Neural Network Approach" (EMNLP 2025)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class SignGNN(torch.nn.Module):
    """
    Graph Neural Network for processing temporal signing sequences.

    Architecture:
        - 3 GCN layers with LeakyReLU activation
        - Batch normalization after each layer
        - Dropout for regularization
        - Global mean pooling for sequence-level embeddings

    The graph structure includes:
        - Spatial edges: anatomical connections within each frame
        - Temporal edges: connections across consecutive frames

    Args:
        input_dim (int): Dimension of input node features (default: 3 for x,y,z coordinates)
        hidden_dim (int): Dimension of hidden layers (default: 64)
        embedding_dim (int): Dimension of output embeddings (default: 32)
    """

    def __init__(self, input_dim=3, hidden_dim=64, embedding_dim=32):
        super(SignGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)

        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            batch (torch.Tensor): Batch assignment for each node [num_nodes]

        Returns:
            torch.Tensor: Graph-level embeddings [batch_size, embedding_dim]
        """
        # First GNN layer - process local keypoint features
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second GNN layer - aggregate finger-level patterns
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Third GNN layer - create hand-level features
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        # Global pooling to get sequence-level embedding
        out = global_mean_pool(x, batch)

        return out

    def get_embedding_dim(self):
        """Return the embedding dimension."""
        return self.embedding_dim

    def save_checkpoint(self, path, epoch, optimizer, scheduler, val_loss, config):
        """
        Save model checkpoint with training state.

        Args:
            path (str): Path to save checkpoint
            epoch (int): Current epoch number
            optimizer: Optimizer state
            scheduler: Scheduler state
            val_loss (float): Validation loss
            config (dict): Training configuration
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'config': config,
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_from_checkpoint(cls, path, device='cpu'):
        """
        Load model from checkpoint.

        Args:
            path (str): Path to checkpoint file
            device (str): Device to load model on

        Returns:
            tuple: (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device)
        model_config = checkpoint.get('model_config', {
            'input_dim': 3,
            'hidden_dim': 64,
            'embedding_dim': 32
        })

        model = cls(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            embedding_dim=model_config['embedding_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint
