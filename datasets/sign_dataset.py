#!/usr/bin/env python3
"""
Dataset for SignGNN: Temporal Sign Language Sequences

Processes PopSign dataset parquet files into graph representations with both
spatial (anatomical) and temporal (across-frame) edges.

Reference: "Improving Handshape Representations for Sign Language Processing:
           A Graph Neural Network Approach" (EMNLP 2025)
"""

import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """
    PyTorch Geometric dataset for temporal sign language sequences.

    Each sample is a graph representing a complete sign with:
        - Nodes: hand landmarks across all frames (21 landmarks × T frames)
        - Spatial edges: anatomical connections within each frame
        - Temporal edges: connections between same landmarks across frames

    Args:
        root (str): Root directory containing PopSign parquet files
        split (str): 'train' or 'val'
        train_val_split (float): Train/val split ratio (default: 0.8)
        max_samples (int, optional): Limit number of samples for testing
    """

    def __init__(self, root: str, split: str = 'train',
                 train_val_split: float = 0.8, max_samples: Optional[int] = None):
        self.root = Path(root).resolve()
        self.split = split
        self.train_val_split = train_val_split
        self.max_samples = max_samples
        self.df = None
        self.valid_indices = []
        self.sign_to_idx = None
        self.min_frames = 1
        self.min_landmarks = 15

        super().__init__(str(self.root))
        self._load_metadata()
        self._split_data()
        self._validate_samples()

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

    def _load_metadata(self):
        """Load train.csv metadata from PopSign."""
        csv_path = self.root / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if self.max_samples:
            self.df = self.df.head(self.max_samples)

        # Create sign to index mapping
        unique_signs = sorted(self.df["sign"].unique())
        self.sign_to_idx = {sign: idx for idx, sign in enumerate(unique_signs)}
        logger.info(f"Loaded {len(self.df)} entries with {len(unique_signs)} unique signs")

    def _split_data(self):
        """Perform train-val split."""
        np.random.seed(42)
        indices = np.arange(len(self.df))
        np.random.shuffle(indices)

        split_idx = int(len(indices) * self.train_val_split)
        if self.split == 'train':
            self.df = self.df.iloc[indices[:split_idx]].reset_index(drop=True)
        elif self.split == 'val':
            self.df = self.df.iloc[indices[split_idx:]].reset_index(drop=True)

        logger.info(f"Split '{self.split}' contains {len(self.df)} samples")

    def _validate_samples(self):
        """Validate file paths and filter valid indices."""
        for idx, row in self.df.iterrows():
            try:
                clean_path = row["path"].replace('train_landmark_files/', '')
                file_path = self.root / clean_path

                if file_path.exists():
                    self.valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Error validating sample {idx}: {str(e)}")

        logger.info(f"Validation completed: {len(self.valid_indices)} valid samples")

    def len(self):
        return len(self.valid_indices)

    def get(self, idx: int) -> Data:
        """Fetch a sample by index."""
        try:
            row = self.df.iloc[self.valid_indices[idx]]
            clean_path = row["path"].replace('train_landmark_files/', '')
            file_path = self.root / clean_path

            # Load parquet file
            df = pd.read_parquet(file_path)
            if df.empty:
                return self._create_dummy_data(self.sign_to_idx[row["sign"]])

            sign_idx = self.sign_to_idx[row["sign"]]
            return self._process_frames(df, sign_idx)

        except Exception as e:
            logger.error(f"Error fetching sample {idx}: {str(e)}")
            return self._create_dummy_data(0)

    def _process_frames(self, df: pd.DataFrame, sign: int) -> Data:
        """Process frames and create graph with temporal + spatial edges."""
        try:
            right_hand = df[df['type'] == 'right_hand']
            unique_frames = sorted(right_hand['frame'].unique())

            frames = []
            for frame_num in unique_frames:
                frame_data = right_hand[right_hand['frame'] == frame_num]

                if len(frame_data) < self.min_landmarks:
                    continue

                # Sort by landmark index and get coordinates
                frame_data = frame_data.sort_values("landmark_index")
                coords = frame_data[['x', 'y', 'z']].values

                # Pad or trim to 21 landmarks
                if len(coords) < 21:
                    padding = np.zeros((21 - len(coords), 3))
                    coords = np.vstack([coords, padding])
                elif len(coords) > 21:
                    coords = coords[:21]

                # Normalize
                normalized = self._normalize_coordinates(coords)
                if normalized is not None and np.all(np.isfinite(normalized)):
                    frames.append(normalized)

            if len(frames) < self.min_frames:
                return self._create_dummy_data(sign)

            # Convert to tensors
            x = torch.tensor(np.stack(frames), dtype=torch.float).reshape(-1, 3)
            y = torch.tensor([sign], dtype=torch.long)

            # Create edge indices
            edge_index = self._create_edges(len(frames))

            return Data(x=x, edge_index=edge_index, y=y)

        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            return self._create_dummy_data(sign)

    def _normalize_coordinates(self, coords: np.ndarray) -> Optional[np.ndarray]:
        """Normalize coordinates relative to wrist with numerical stability."""
        try:
            wrist = coords[0]
            centered = coords - wrist

            distances = np.linalg.norm(centered, axis=1)
            scale = np.clip(distances.max(), 1e-6, None)

            normalized = centered / scale
            if not np.all(np.isfinite(normalized)):
                normalized = np.nan_to_num(normalized, 0.0)

            return normalized

        except Exception as e:
            logger.error(f"Error in normalization: {str(e)}")
            return None

    def _create_edges(self, num_frames: int) -> torch.Tensor:
        """
        Create temporal and spatial edges for hand landmarks.

        Returns edge_index tensor combining:
            - Spatial edges: anatomical connections within each frame
            - Temporal edges: connections across consecutive frames
        """
        # Define hand landmark connections (MediaPipe 21-point hand model)
        spatial_edges = [
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
            # Palm connections
            [5, 9], [9, 13], [13, 17]
        ]

        all_edges = []
        for frame in range(num_frames):
            # Add spatial edges for current frame
            offset = frame * 21
            for edge in spatial_edges:
                all_edges.append([edge[0] + offset, edge[1] + offset])
                all_edges.append([edge[1] + offset, edge[0] + offset])  # Bidirectional

            # Add temporal edges to next frame
            if frame < num_frames - 1:
                for landmark in range(21):
                    curr_idx = frame * 21 + landmark
                    next_idx = (frame + 1) * 21 + landmark
                    all_edges.append([curr_idx, next_idx])
                    all_edges.append([next_idx, curr_idx])  # Bidirectional

        return torch.tensor(all_edges, dtype=torch.long).t().contiguous()

    def _create_dummy_data(self, sign: int = 0) -> Data:
        """Create a valid dummy Data object for error cases."""
        dummy_frame = np.zeros((21, 3))
        x = torch.tensor([dummy_frame], dtype=torch.float).reshape(-1, 3)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        y = torch.tensor([sign], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)
