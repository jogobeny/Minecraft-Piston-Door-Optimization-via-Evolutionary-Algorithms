from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Block:
    _namespaced_id: str  # https://minecraft.wiki/w/Identifier
    facing: Literal["down", "east", "north", "south", "up", "west"] | None = None

    @property
    def namespaced_id(self):
        if self.facing is not None:
            return f"{self._namespaced_id}[facing={self.facing}]"
        return self._namespaced_id


AIR = Block("minecraft:air")
STONE = Block("minecraft:stone")
SLIME_BLOCK = Block("minecraft:slime_block")

REDSTONE_DUST = Block("minecraft:redstone_wire")
REDSTONE_BLOCK = Block("minecraft:redstone_block")

STICKY_PISTON_D = Block("minecraft:sticky_piston", facing="down")
STICKY_PISTON_E = Block("minecraft:sticky_piston", facing="east")
STICKY_PISTON_N = Block("minecraft:sticky_piston", facing="north")
STICKY_PISTON_S = Block("minecraft:sticky_piston", facing="south")
STICKY_PISTON_U = Block("minecraft:sticky_piston", facing="up")
STICKY_PISTON_W = Block("minecraft:sticky_piston", facing="west")

SOLID_BLOCKS = [
    STONE,
    SLIME_BLOCK,
    # REDSTONE_BLOCK,
    STICKY_PISTON_D,
    STICKY_PISTON_E,
    STICKY_PISTON_N,
    STICKY_PISTON_S,
    STICKY_PISTON_U,
    STICKY_PISTON_W,
]
