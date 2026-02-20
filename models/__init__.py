"""
Models for Handshape Recognition using Graph Neural Networks

This package contains the core model architectures from:
"Improving Handshape Representations for Sign Language Processing:
 A Graph Neural Network Approach" (EMNLP 2025)
"""

from .sign_gnn import SignGNN
from .handshape_gnn import HandshapeGNN
from .triple_classifier import TripleStreamClassifier
from .baseline import BaselineMLP

__all__ = [
    'SignGNN',
    'HandshapeGNN',
    'TripleStreamClassifier',
    'BaselineMLP'
]
