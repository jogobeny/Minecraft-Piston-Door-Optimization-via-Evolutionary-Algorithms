from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """
    If some blocks have facing variants, the ID is expanded.
    For example:
      - 29: sticky piston
      - 2901: sticky piston facing north
      - 2902: sticky piston facing south
      - 2903: sticky piston facing west
      - 2904: sticky piston facing east

    .. seealso:: https://minecraft-ids.grahamedgecombe.com
    """

    id: int
    namespaced_id: str


AIR = Block(0, "minecraft:air")
STONE = Block(1, "minecraft:stone")
SLIME_BLOCK = Block(165, "minecraft:slime_block")

REDSTONE_WIRE = Block(55, "minecraft:redstone_wire")
REDSTONE_BLOCK = Block(152, "minecraft:redstone_block")

STICKY_PISTON_N = Block(2901, "minecraft:sticky_piston[facing=north]")
STICKY_PISTON_S = Block(2902, "minecraft:sticky_piston[facing=south]")
STICKY_PISTON_W = Block(2903, "minecraft:sticky_piston[facing=west]")
STICKY_PISTON_E = Block(2904, "minecraft:sticky_piston[facing=east]")


SOLID_BLOCKS = [
    STONE,
    SLIME_BLOCK,
    REDSTONE_BLOCK,
    STICKY_PISTON_N,
    STICKY_PISTON_S,
    STICKY_PISTON_W,
    STICKY_PISTON_E,
]
