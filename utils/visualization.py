#!/usr/bin/env python3
"""
Visualization Utilities

Functions for plotting confusion matrices, training curves, and analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          output_path: Path,
                          title: str = 'Confusion Matrix',
                          figsize: tuple = (20, 20)):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix [num_classes, num_classes]
        class_names: List of class names
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: List[Dict],
                          output_path: Path,
                          metrics: List[str] = ['loss', 'accuracy']):
    """
    Plot training curves.

    Args:
        history: List of training history dictionaries
        output_path: Path to save figure
        metrics: Metrics to plot
    """
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history[0]:
            train_values = [h[train_key] for h in history]
            axes[idx].plot(epochs, train_values, label=f'Train {metric}', marker='o')

        if val_key in history[0]:
            val_values = [h[val_key] for h in history]
            axes[idx].plot(epochs, val_values, label=f'Val {metric}', marker='s')

        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].set_title(f'{metric.capitalize()} over Epochs', fontsize=14)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_distribution(class_counts: Dict[str, int],
                            output_path: Path,
                            title: str = 'Handshape Distribution',
                            top_n: int = None):
    """
    Plot class distribution histogram.

    Args:
        class_counts: Dictionary mapping class names to counts
        output_path: Path to save figure
        title: Plot title
        top_n: Only show top N classes (None for all)
    """
    # Sort by count
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_items = sorted_items[:top_n]

    classes, counts = zip(*sorted_items)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(classes)), counts, color='steelblue', edgecolor='black')

    # Color the top 3 differently
    for i in range(min(3, len(bars))):
        bars[i].set_color('coral')

    plt.xlabel('Handshape', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(models_metrics: Dict[str, Dict],
                           output_path: Path,
                           metric: str = 'f1'):
    """
    Compare a specific metric across different models.

    Args:
        models_metrics: Dict of {model_name: {class_name: {metric: value}}}
        output_path: Path to save figure
        metric: Metric to compare
    """
    # Get all class names
    first_model = list(models_metrics.values())[0]
    class_names = sorted(first_model.keys())

    # Prepare data
    model_names = list(models_metrics.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(16, 6))

    for idx, model_name in enumerate(model_names):
        values = [models_metrics[model_name][cls][metric] for cls in class_names]
        offset = (idx - len(model_names) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=model_name, alpha=0.8)

    ax.set_xlabel('Handshape', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=12)
    ax.set_title(f'{metric.capitalize()} Comparison Across Models', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
