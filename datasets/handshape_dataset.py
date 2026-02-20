#!/usr/bin/env python3
"""
Dataset for HandshapeGNN: Static Handshape Frames

Processes merged PopSign+ASL-LEX JSON data, selecting low-motion frames
for static handshape analysis with anatomically-informed graph structures.

Reference: "Improving Handshape Representations for Sign Language Processing:
           A Graph Neural Network Approach" (EMNLP 2025)
"""

import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Optional, List, Dict
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HandshapeDataset(Dataset):
    """
    PyTorch Geometric dataset for static handshape recognition.

    Each sample is a graph representing a single low-motion frame with:
        - Nodes: 21 hand landmarks (MediaPipe format)
        - Edges: Anatomically-informed bidirectional connections

    Args:
        json_path (str): Path to merged PopSign+ASL-LEX JSON file
        split (str): 'train' or 'val'
        val_ratio (float): Validation split ratio (default: 0.2)
    """

    def __init__(self, json_path: str, split: str = 'train', val_ratio: float = 0.2):
        self.json_path = json_path
        self.split = split
        self.val_ratio = val_ratio
        self.samples = []
        self.handshape_to_idx = {}
        self.idx_to_handshape = {}

        super().__init__('')
        self._load_data()
        self._create_splits()
        logger.info(f"Initialized {split} dataset with {len(self.samples)} samples")

    @property
    def processed_dir(self) -> str:
        return ''

    @property
    def raw_dir(self) -> str:
        return ''

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        """
        Get a single sample as a PyG Data object.

        Selects the frame with minimum motion (lowest keypoint displacement)
        as the most representative static handshape configuration.
        """
        sample = self.samples[idx]
        handshape_idx = self.handshape_to_idx[sample['handshape']]

        # Get landmarks from preprocessed data
        landmarks = sample['processed_landmarks']
        graph_data = self._create_hand_graph(landmarks)
        graph_data.handshape = torch.tensor([handshape_idx], dtype=torch.long)

        return graph_data

    def _load_data(self):
        """Load and preprocess JSON data, filtering invalid samples."""
        logger.info("Loading and preprocessing data...")
        with open(self.json_path) as f:
            data = json.load(f)

        # Filter and preprocess samples
        valid_samples = []
        for entry in data:
            if 'handshape' not in entry or 'landmarks' not in entry or entry['handshape'] is None:
                continue

            # Find low-motion frame
            valid_landmarks = self._select_low_motion_frame(entry['landmarks'])

            if valid_landmarks is not None:
                entry['processed_landmarks'] = valid_landmarks
                valid_samples.append(entry)

        logger.info(f"Found {len(valid_samples)} valid samples out of {len(data)} total")

        # Create label mappings
        unique_handshapes = sorted(list(set(s['handshape'] for s in valid_samples)))
        self.handshape_to_idx = {hs: idx for idx, hs in enumerate(unique_handshapes)}
        self.idx_to_handshape = {idx: hs for hs, idx in self.handshape_to_idx.items()}

        # Log class distribution
        class_counts = defaultdict(int)
        for entry in valid_samples:
            class_counts[entry['handshape']] += 1

        logger.info(f"Found {len(unique_handshapes)} unique handshapes")
        logger.info("Class distribution:")
        for handshape, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {handshape}: {count} samples")

        self.samples = valid_samples

    def _create_splits(self):
        """Create train/validation splits."""
        if self.val_ratio == 0.0:
            logger.info(f"Skipping split, using all {len(self.samples)} samples")
            return

        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * (1 - self.val_ratio))

        if self.split == 'train':
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]

    def _select_low_motion_frame(self, frames: List[Dict]) -> Optional[torch.Tensor]:
        """
        Select the frame with minimum motion as the most representative
        static handshape configuration.

        Args:
            frames: List of frame dictionaries with landmark data

        Returns:
            Normalized landmarks tensor or None if no valid frame
        """
        valid_frames = []
        motion_scores = []

        for i, frame in enumerate(frames):
            landmarks = self._process_landmarks(frame)
            if landmarks is None:
                continue

            valid_frames.append(landmarks)

            # Calculate motion score (displacement from next frame)
            if i < len(frames) - 1:
                next_landmarks = self._process_landmarks(frames[i + 1])
                if next_landmarks is not None:
                    motion = torch.norm(landmarks - next_landmarks).item()
                    motion_scores.append(motion)
                else:
                    motion_scores.append(float('inf'))
            else:
                motion_scores.append(float('inf'))

        if not valid_frames:
            return None

        # Select frame with minimum motion
        min_motion_idx = np.argmin(motion_scores) if motion_scores else 0
        return valid_frames[min_motion_idx]

    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates relative to wrist position."""
        wrist = coords[0]
        centered = coords - wrist
        scale = np.clip(np.linalg.norm(centered, axis=1).max(), 1e-6, None)
        normalized = centered / scale
        return normalized

    def _create_hand_graph(self, landmarks: torch.Tensor) -> Data:
        """
        Create graph with anatomically-informed bidirectional edges.

        Edge types:
            - Sequential: finger joint connections
            - Cross-finger: connections between adjacent fingers
            - Palm-centered: wrist to finger base connections
            - Diagonal palm: connections across palm
        """
        edges = [
            # Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            # Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            # Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            # Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            # Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
            # Palm connections (across fingers)
            [5, 9], [9, 13], [13, 17],
            # Diagonal palm connections
            [1, 5], [5, 9], [9, 13], [13, 17]
        ]

        # Create bidirectional edges
        edges = edges + [[j, i] for i, j in edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=landmarks, edge_index=edge_index)

    def _process_landmarks(self, frame: Dict) -> Optional[torch.Tensor]:
        """Process and normalize landmarks from a single frame."""
        if all(key in frame for key in ['x_right_hand', 'y_right_hand', 'z_right_hand']):
            coords = []
            for x, y, z in zip(frame['x_right_hand'],
                             frame['y_right_hand'],
                             frame['z_right_hand']):
                if not (pd.isna(x) or pd.isna(y) or pd.isna(z)):
                    coords.append([x, y, z])

            if len(coords) == 21:  # All landmarks present
                coords = np.array(coords)
                normalized = self._normalize_coordinates(coords)
                return torch.tensor(normalized, dtype=torch.float32)

        return None
