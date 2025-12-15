import asyncio
import logging
import math
import random
import time

import coloredlogs
import numpy as np
from deap import algorithms, base, creator, tools

from ea import blocks
from ea.blocks import SOLID_BLOCKS

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)

coordinates_type = [("x", int), ("y", int), ("z", int)]

BOUNDS = np.array([(5, 3, 5)], dtype=coordinates_type)

BASE_POSITION = np.array([(0, -60, 0)], dtype=coordinates_type)
OFFSET = np.array([(4, 0, 4)], dtype=coordinates_type)

# DOOR_BASE_POSITION = np.array([(4, 0, 2), (4, 1, 2)], dtype=coordinates_type)
# DOOR_TARGET_POSITIONS = np.array([(5, 0, 2), (5, 1, 2)], dtype=coordinates_type)
DOOR_TARGET_POSITIONS = np.array(
    [
        # Top
        [(0, 0, -1), (0, 1, -1)],
        [(1, 0, -1), (1, 1, -1)],
        [(2, 0, -1), (2, 1, -1)],
        [(3, 0, -1), (3, 1, -1)],
        [(4, 0, -1), (4, 1, -1)],
        # Bottom
        [(0, 0, BOUNDS["z"].item()), (0, 1, BOUNDS["z"].item())],
        [(1, 0, BOUNDS["z"].item()), (1, 1, BOUNDS["z"].item())],
        [(2, 0, BOUNDS["z"].item()), (2, 1, BOUNDS["z"].item())],
        [(3, 0, BOUNDS["z"].item()), (3, 1, BOUNDS["z"].item())],
        [(4, 0, BOUNDS["z"].item()), (4, 1, BOUNDS["z"].item())],
        # Left
        [(-1, 0, 0), (-1, 1, 0)],
        [(-1, 0, 1), (-1, 1, 1)],
        [(-1, 0, 2), (-1, 1, 2)],
        [(-1, 0, 3), (-1, 1, 3)],
        [(-1, 0, 4), (-1, 1, 4)],
        # Right
        [(BOUNDS["x"].item(), 0, 0), (BOUNDS["x"].item(), 1, 0)],
        [(BOUNDS["x"].item(), 0, 1), (BOUNDS["x"].item(), 1, 1)],
        [(BOUNDS["x"].item(), 0, 2), (BOUNDS["x"].item(), 1, 2)],
        [(BOUNDS["x"].item(), 0, 3), (BOUNDS["x"].item(), 1, 3)],
        [(BOUNDS["x"].item(), 0, 4), (BOUNDS["x"].item(), 1, 4)],
    ],
    dtype=coordinates_type,
)


TORCH_POSITION = np.array([(5, 0, 4)], dtype=coordinates_type)

POSSIBLE_BLOCKS = [blocks.AIR, blocks.REDSTONE_WIRE] + SOLID_BLOCKS


def get_global_position(x: int, y: int, z: int, offset: np.ndarray):
    return (
        BASE_POSITION.view(int) + OFFSET.view(int) + offset.view(int) + np.array([(x, y, z)])
    ).ravel()


def create_individual():
    return np.array(
        [
            [
                [random.choice(POSSIBLE_BLOCKS) for x in range(BOUNDS["x"].item())]
                for z in range(BOUNDS["z"].item())
            ]
            for y in range(BOUNDS["y"].item())
        ]
    )


def mutate(individual, indpb):
    for y in range(individual.shape[0]):
        for z in range(individual.shape[1]):
            for x in range(individual.shape[2]):
                if random.random() < indpb:
                    original_block = individual[y, z, x]
                    new_block = random.choice(POSSIBLE_BLOCKS)

                    if new_block == blocks.REDSTONE_WIRE:
                        if y > 0:
                            block_below = individual[y - 1, z, x]
                            if block_below in SOLID_BLOCKS:
                                individual[y, z, x] = new_block
                        elif y == 0:
                            individual[y, z, x] = new_block
                    else:
                        individual[y, z, x] = new_block

    for y in range(1, individual.shape[0]):
        for z in range(individual.shape[1]):
            for x in range(individual.shape[2]):
                block = individual[y, z, x]
                if block == blocks.REDSTONE_WIRE:
                    block_below = individual[y - 1, z, x]
                    if block_below not in SOLID_BLOCKS:
                        individual[y, z, x] = blocks.AIR

    return (individual,)


def cxUniformVoxel(ind1, ind2, indpb):
    mask = np.random.random(ind1.shape) < indpb
    ind1[mask], ind2[mask] = ind2[mask], ind1[mask]
    return ind1, ind2


toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


def init_numpy_individual(icls, content_func):
    return icls(content_func())


toolbox.register("individual", init_numpy_individual, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", cxUniformVoxel, indpb=0.15)
toolbox.register("mutate", mutate, indpb=0.08)
toolbox.register("select", tools.selTournament, tournsize=3)


class MinecraftBatchEvaluator:

    def __init__(self, server_context, population_size: int):
        self.server_context = server_context
        self.population_size = population_size
        self.grid = self._best_grid(population_size)

    def _best_grid(self, population_size: int):
        root = math.isqrt(population_size)
        for n in range(root, 0, -1):
            if population_size % n == 0:
                return n, population_size // n

    async def clear_area(self):
        bx, by, bz = BASE_POSITION.view(int)
        dx = self.grid[0] * 16 - 1
        dz = self.grid[1] * 16 - 1
        logger.debug(f"Clearing area {bx} {by} {bz} {bx + dx} {by + 4} {bz + dz}")
        await self.server_context.clear_area(bx, by, bz, bx + dx, by + 4, bz + dz)

    async def __aenter__(self):
        await self.clear_area()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    def chunk_offset(self, i: int):
        row = i // self.grid[1]
        col = i % self.grid[1]
        return np.array([(row * 16, 0, col * 16)], dtype=coordinates_type)

    async def build_individual(self, individual, offset):
        # for x, y, z in DOOR_BASE_POSITION:
        #     gx, gy, gz = get_global_position(x, y, z, offset)
        #     await self.server_context.set_block(gx, gy, gz, blocks.STONE)

        for y in range(individual.shape[0]):
            for z in range(individual.shape[1]):
                for x in range(individual.shape[2]):
                    block = individual[y, z, x]
                    if block == blocks.AIR:
                        continue
                    gx, gy, gz = get_global_position(x, y, z, offset)
                    await self.server_context.set_block(gx, gy, gz, block)

    async def check_door_state(self, positions, offset):
        blocks = 0
        for tx, ty, tz in positions:
            gx, gy, gz = get_global_position(tx, ty, tz, offset)
            response = await self.server_context.run_command(
                f"execute if block {gx} {gy} {gz} minecraft:stone"
            )
            if response == "Test passed":
                blocks += 1

        return blocks

    async def evaluate(self, population):
        await self.clear_area()

        for i, individual in enumerate(population):
            offset = self.chunk_offset(i)
            await self.build_individual(individual, offset)

        await asyncio.sleep(0.25)

        # scores_open = []
        # for i, individual in enumerate(population):
        #     offset = self.chunk_offset(i)
        #     scores_open.append(await self.check_door_state(DOOR_BASE_POSITION, offset))

        for i, _ in enumerate(population):
            offset = self.chunk_offset(i)
            await self.server_context.set_block(
                *get_global_position(*(TORCH_POSITION.item()), offset), "minecraft:redstone_torch"
            )

        await asyncio.sleep(0.5)

        await asyncio.sleep(10)

        fitnesses = []
        for i, _ in enumerate(population):
            offset = self.chunk_offset(i)

            counts = []
            for j in range(len(DOOR_TARGET_POSITIONS)):
                pair = DOOR_TARGET_POSITIONS[j]
                blocks_in_pair = 0
                for tx, ty, tz in pair:
                    gx, gy, gz = get_global_position(tx, ty, tz, offset)
                    print(gx, gy, gz, "koukám na")
                    response = await self.server_context.run_command(
                        f"execute if block {gx} {gy} {gz} minecraft:stone"
                    )
                    print(response)
                    if response == "Test passed":
                        blocks_in_pair += 1
                counts.append(blocks_in_pair)

            max_segment_score = max(counts)
            total_blocks_found = sum(counts)

            if max_segment_score == 0:
                score = 0
            else:
                base_score = 10 if max_segment_score == 1 else 50
                noise_blocks = total_blocks_found - max_segment_score
                penalty = noise_blocks * 5
                score = max(0, base_score - penalty)

            print(score)

            fitnesses.append((score,))

        # scores_closed = []
        # for i, _ in enumerate(population):
        #     offset = self.chunk_offset(i)
        #     scores_closed.append(await self.check_door_state(DOOR_TARGET_POSITIONS, offset))

        # fitnesses = []
        # for s in scores_closed:
        #     fitnesses.append((s,))

        return fitnesses


async def run(server_context):
    POPULATION_SIZE = 1
    NGEN = 1

    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    async with MinecraftBatchEvaluator(server_context, POPULATION_SIZE) as evaluator:
        logger.info(f"Generation {0}")

        fitnesses = await evaluator.evaluate(population)
        for individual, fitness in zip(population, fitnesses):
            individual.fitness.values = fitness

        record = stats.compile(population)
        hof.update(population)

        for i in range(1, NGEN):
            logger.info(f"Generation {i}")

            offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(x) for x in offspring]

            for j in range(1, len(offspring), 2):
                if np.random.random() < 0.6:
                    toolbox.mate(offspring[j - 1], offspring[j])
                    del offspring[j - 1].fitness.values
                    del offspring[j].fitness.values

            for j in range(len(offspring)):
                if np.random.random() < 0.3:
                    toolbox.mutate(offspring[j])
                    del offspring[j].fitness.values

            invalid_population = [x for x in offspring if not x.fitness.valid]

            if invalid_population:
                fitnesses = await evaluator.evaluate(invalid_population)
                for individual, fitness in zip(invalid_population, fitnesses):
                    individual.fitness.values = fitness

            population[:] = offspring
            hof.update(population)
            record = stats.compile(population)
