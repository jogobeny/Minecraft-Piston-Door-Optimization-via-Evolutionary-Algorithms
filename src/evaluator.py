# import logging
# import math

# import coloredlogs
# import numpy as np

# from .server import MinecraftServerContext

# logger = logging.getLogger(__name__)
# coloredlogs.install(level="INFO", logger=logger)


# class MinecraftBatchEvaluator:

#     def __init__(self, server_context: MinecraftServerContext, population_size: int):
#         self.server_context = server_context
#         self.population_size = population_size

#         for n in range(math.isqrt(population_size), 0, -1):
#             if population_size % n == 0:
#                 self.grid = n, population_size // n

#     async def clear_area(self):
#         bx, by, bz = self.server_context.base_position
#         dx = self.grid[0] * 16 - 1
#         dz = self.grid[1] * 16 - 1
#         await self.server_context.clear_area(bx, by, bz, bx + dx, by + 4, bz + dz)

#     async def __aenter__(self):
#         await self.clear_area()
#         return self

#     async def __aexit__(self, exc_type, exc_value, traceback):
#         pass

#     def chunk_offset(self, i: int):
#         row = i // self.grid[1]
#         col = i % self.grid[1]
#         return np.array([(row * 16, 0, col * 16)])

#     async def build_individual(self, individual, offset):
#         for y in range(individual.shape[0]):
#             for z in range(individual.shape[1]):
#                 for x in range(individual.shape[2]):
#                     block = individual[y, z, x]
#                     if block == blocks.AIR:
#                         continue
#                     gx, gy, gz = get_global_position(x, y, z, offset)
#                     await self.server_context.set_block(gx, gy, gz, block)

#     async def check_door_state(self, positions, offset):
#         blocks = 0
#         for tx, ty, tz in positions:
#             gx, gy, gz = get_global_position(tx, ty, tz, offset)
#             response = await self.server_context.run_command(
#                 f"execute if block {gx} {gy} {gz} minecraft:stone"
#             )
#             if response[0] == "Test passed":
#                 blocks += 1

#         return blocks

#     async def evaluate(self, population):
#         await self.clear_area()

#         for i, individual in enumerate(population):
#             offset = self.chunk_offset(i)
#             await self.build_individual(individual, offset)

#         await asyncio.sleep(0.25)

#         for i, individual in enumerate(population):
#             offset = self.chunk_offset(i)
#             await self.server_context.set_block(
#                 *get_global_position(*(individual.torch_position.item()), offset),
#                 "minecraft:redstone_torch",
#             )

#         await asyncio.sleep(0.5)

#         fitnesses = []
#         for i, individual in enumerate(population):
#             offset = self.chunk_offset(i)

#             counts = []
#             for j in range(len(DOOR_TARGET_POSITIONS)):
#                 pair = DOOR_TARGET_POSITIONS[j]
#                 blocks_in_pair = 0
#                 for tx, ty, tz in pair:
#                     gx, gy, gz = get_global_position(tx, ty, tz, offset)
#                     response = await self.server_context.run_command(
#                         f"execute if block {gx} {gy} {gz} minecraft:stone"
#                     )
#                     if response[0] == "Test passed":
#                         blocks_in_pair += 1
#                 counts.append(blocks_in_pair)

#             max_segment_score = max(counts)
#             total_blocks_found = sum(counts)

#             if max_segment_score == 0:
#                 score = 0
#             else:
#                 base_score = 10 if max_segment_score == 1 else 50
#                 noise_blocks = total_blocks_found - max_segment_score
#                 penalty = noise_blocks * 5
#                 score = max(0, base_score - penalty)

#             fitnesses.append((score,))

#         return fitnesses
