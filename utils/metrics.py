#!/usr/bin/env python3
"""
Evaluation Metrics and Biomechanical Analysis

Implements metrics for handshape classification evaluation and biomechanical
analysis of handshape configurations.

Reference: "Improving Handshape Representations for Sign Language Processing:
           A Graph Neural Network Approach" (EMNLP 2025)
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import Dict, Tuple


def compute_classification_metrics(outputs: torch.Tensor,
                                   labels: torch.Tensor,
                                   idx_to_class: Dict[int, str]) -> Tuple[Dict, np.ndarray, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        outputs: Model logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        idx_to_class: Mapping from class indices to class names

    Returns:
        Tuple of (per_class_metrics, confusion_matrix, accuracy)
    """
    _, predicted = outputs.max(1)

    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    # Overall accuracy
    total = labels.size
    correct = (predicted == labels).sum()
    accuracy = 100. * correct / total

    # Confusion matrix
    cm = confusion_matrix(labels, predicted)

    # Per-class metrics
    metrics = {}
    for idx in range(len(idx_to_class)):
        label_mask = labels == idx
        pred_mask = predicted == idx

        true_positives = np.sum(predicted[label_mask] == idx)
        false_positives = np.sum(pred_mask) - true_positives
        false_negatives = np.sum(label_mask) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[idx_to_class[idx]] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(label_mask))
        }

    return metrics, cm, accuracy


def calculate_finger_independence(landmarks: np.ndarray) -> float:
    """
    Calculate finger independence score.

    Measures the variation in joint angles across different fingers,
    indicating how independently fingers are configured.

    Args:
        landmarks: Hand landmarks [21, 3]

    Returns:
        Finger independence score (higher = more independent)
    """
    # Define joint groups
    metacarpo = [5, 9, 13, 17]  # Metacarpophalangeal joints
    proximal = [6, 10, 14, 18]  # Proximal interphalangeal joints
    distal = [7, 11, 15, 19]    # Distal interphalangeal joints

    joint_groups = [metacarpo, proximal, distal]

    total_score = 0
    for group in joint_groups:
        angles = []
        for joint_idx in group:
            if joint_idx < len(landmarks):
                # Calculate angle (simplified: distance from wrist)
                angle = np.linalg.norm(landmarks[joint_idx] - landmarks[0])
                angles.append(angle)

        # Calculate variation across fingers
        if len(angles) > 1:
            score = np.sum([abs(angles[i] - angles[j])
                           for i in range(len(angles))
                           for j in range(i + 1, len(angles))])
            total_score += score

    return total_score


def calculate_thumb_effort(landmarks: np.ndarray,
                           resting_position: np.ndarray = None) -> float:
    """
    Calculate thumb effort score.

    Measures deviation from resting position, indicating articulatory effort.

    Args:
        landmarks: Hand landmarks [21, 3]
        resting_position: Reference resting hand configuration [21, 3]

    Returns:
        Thumb effort score (higher = more effort)
    """
    thumb_joints = [1, 2, 3, 4]

    if resting_position is None:
        # Use default resting position (simplified)
        resting_position = np.zeros_like(landmarks)

    effort = 0
    for joint_idx in thumb_joints:
        if joint_idx < len(landmarks):
            diff = np.linalg.norm(landmarks[joint_idx] - resting_position[joint_idx])
            effort += diff

    return effort / len(thumb_joints)


def calculate_handshape_distance(landmarks1: np.ndarray,
                                 landmarks2: np.ndarray) -> float:
    """
    Calculate perceptual distance between two handshapes.

    Args:
        landmarks1: First hand configuration [21, 3]
        landmarks2: Second hand configuration [21, 3]

    Returns:
        Distance score (higher = more different)
    """
    # Calculate average joint-wise distance
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    return np.mean(distances)


def contrastive_loss(embeddings: torch.Tensor,
                     batch,
                     temperature: float = 0.1) -> torch.Tensor:
    """
    Compute contrastive loss for training GNNs.

    Encourages embeddings of same class to be similar and different classes
    to be dissimilar.

    Args:
        embeddings: Model embeddings [batch_size, embedding_dim]
        batch: Batch object containing labels
        temperature: Temperature parameter for scaling

    Returns:
        Contrastive loss value
    """
    device = embeddings.device

    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / temperature

    # Extract labels
    if hasattr(batch, 'y'):
        labels = batch.y.view(-1)
    elif hasattr(batch, 'handshape'):
        labels = batch.handshape.view(-1)
    else:
        raise ValueError("Batch must have 'y' or 'handshape' attribute")

    # Create positive mask (same label)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    neg_mask = ~pos_mask

    # Remove self-comparisons
    mask_self = ~torch.eye(embeddings.size(0), dtype=torch.bool, device=device)
    pos_mask = pos_mask & mask_self
    neg_mask = neg_mask & mask_self

    # Handle edge cases
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    # Get positive and negative similarities
    pos_sims = sim_matrix[pos_mask]
    neg_sims = sim_matrix[neg_mask]

    # Combine into logits and labels
    logits = torch.cat([pos_sims, neg_sims])
    targets = torch.cat([
        torch.ones(pos_sims.size(0), device=device),
        torch.zeros(neg_sims.size(0), device=device)
    ])

    # Binary cross-entropy loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    return loss
