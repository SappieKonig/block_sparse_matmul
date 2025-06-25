import torch
from dataclasses import dataclass
from functools import cached_property
import random


@dataclass
class BlockMask:
    mask: torch.Tensor

    @cached_property
    def num_rows(self):
        return self.mask.shape[0]

    @cached_property
    def num_cols(self):
        return self.mask.shape[1]

    @cached_property
    def row_occupancy(self):
        """Amount of non-zero entries in a row"""
        return self.mask.sum(dim=1)[0].item()

    @cached_property
    def col_occupancy(self):
        """Amount of non-zero entries in a column"""
        return self.mask.sum(dim=0)[0].item()

    @cached_property
    def total_blocks(self):
        return self.num_rows * self.row_occupancy

    @cached_property
    def matmul_block_indices(self):
        """Indices are to be compressed over the row dimension, as that's what the reduction would be for A @ B, where A is dense and B is block sparse."""
        indices = torch.zeros((self.col_occupancy, self.num_cols), dtype=torch.int32, device=self.mask.device)
        for col in range(self.num_cols):
            idx = 0
            for row in range(self.num_rows):
                if self.mask[row, col]:
                    indices[idx, col] = row
                    idx += 1
        idx_ranged = torch.arange(self.num_cols).repeat(self.col_occupancy, 1).to(self.mask.device)
        return indices, self._data_indices[indices, idx_ranged]

    @cached_property
    def matmul_block_indices_transposed(self):
        """Indices are to be compressed over the row dimension, as that's what the reduction would be for A @ B, where A is dense and B is block sparse."""
        indices = torch.zeros((self.num_rows, self.row_occupancy), dtype=torch.int32, device=self.mask.device)
        for row in range(self.num_rows):
            idx = 0
            for col in range(self.num_cols):
                if self.mask[row, col]:
                    indices[row, idx] = col
                    idx += 1
        idx_ranged = torch.arange(self.num_rows).view(-1, 1).repeat(1, self.row_occupancy).to(self.mask.device)
        return indices, self._data_indices[idx_ranged, indices]

    def __post_init__(self):
        assert self.mask.ndim == 2
        summed_0 = self.mask.sum(dim=0)
        assert summed_0.eq(summed_0[0]).all()
        summed_1 = self.mask.sum(dim=1)
        assert summed_1.eq(summed_1[0]).all()
        self._data_indices = self._build_data_indices()

    def _build_data_indices(self):
        data_indices = torch.full((self.num_rows, self.num_cols), fill_value=-1, dtype=torch.int32, device=self.mask.device)
        idx = 0
        for col in range(self.num_cols):
            for row in range(self.num_rows):
                if self.mask[row, col]:
                    data_indices[row, col] = idx
                    idx += 1
        return data_indices

    @classmethod
    def regular(cls, size: int, occupancy: int, device: str):
        mask = torch.zeros((size, size), dtype=torch.bool, device=device)
        for _ in range(occupancy):
            while True:
                new_indices = list(range(size))
                random.shuffle(new_indices)
                if any(mask[idx, col] for col, idx in enumerate(new_indices)):
                    continue
                else:
                    for col, idx in enumerate(new_indices):
                        mask[idx, col] = True
                    break
        return cls(mask)

    @classmethod
    def tiled_regular(cls, size: int, occupancy: int, repeat: tuple[int, int], device: str):
        mask = torch.zeros((repeat[0] * size, repeat[1] * size), dtype=torch.bool, device=device)
        for i in range(repeat[0]):
            for j in range(repeat[1]):
                mask[i * size:(i + 1) * size, j * size:(j + 1) * size] = cls.regular(size, occupancy, device).mask
        # shuffle along both dimensions
        row_indices = list(range(repeat[0] * size))
        col_indices = list(range(repeat[1] * size))
        random.shuffle(row_indices)
        random.shuffle(col_indices)
        mask = mask[row_indices, :][:, col_indices]
        return cls(mask)