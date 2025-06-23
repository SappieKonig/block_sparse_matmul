import torch
import random
from dataclasses import dataclass


@dataclass
class Coordinate:
    row: int
    col: int


@dataclass
class Block:
    coord: Coordinate
    index: int


class BlockRegularMatrix:
    def __init__(self, block_size: int, num_blocks: int, k: int, device: str, dtype: torch.dtype):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.k = k
        self.device = device
        self.dtype = dtype
        self.data = torch.randn((block_size, block_size * k * num_blocks), dtype=dtype, device=device)
        self.coordinates = self.create_blueprint()

    def __repr__(self):
        FULL = "██"
        EMPTY = "  "

        example = torch.zeros((self.num_blocks, self.num_blocks))
        for block in self.coordinates:
            example[block.coord.row, block.coord.col] = 1
        example = example.tolist()
        example_string = "\n".join(["".join([FULL if x else EMPTY for x in y]) for y in example])
        return example_string

    def create_blueprint(self):
        blueprint = []
        for i in range(self.k):
            indices = list(range(self.num_blocks))
            while True:
                random.shuffle(indices)
                for used in blueprint:
                    if any(x == y for x, y in zip(used, indices)):
                        break
                else:
                    blueprint.append(indices)
                    break
        coordinates = []
        for l in blueprint:
            for col, row in enumerate(l):
                coordinates.append(Coordinate(row, col))
        # sort by column
        coordinates = sorted(coordinates, key=lambda x: (x.col, x.row))
        blocks = [Block(coord, i) for i, coord in enumerate(coordinates)]
        return blocks

    def as_full(self):
        full = torch.zeros((self.num_blocks * self.block_size, self.num_blocks * self.block_size), dtype=self.dtype, device=self.device)
        for block in self.coordinates:
            idx = block.index
            row, col = block.coord.row, block.coord.col
            row_start = row * self.block_size
            col_start = col * self.block_size
            full[row_start:row_start + self.block_size, col_start:col_start + self.block_size] = self.data[:, idx * self.block_size:(idx + 1) * self.block_size]
        return full