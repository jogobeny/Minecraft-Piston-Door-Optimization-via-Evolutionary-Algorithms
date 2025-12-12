import logging
import random
import time

import coloredlogs
import numpy as np
from deap import algorithms, base, creator, tools

from ea import blocks
from ea.blocks import SOLID_BLOCKS

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)

BOUNDS = {"x": 4, "y": 3, "z": 5}

BASE_POSITION = (0, -60, 0)
OFFSET = (2, 0, 4)

DOOR_BASE_POSITION = [(4, 0, 2), (4, 1, 2)]
DOOR_TARGET_POSITIONS = [(5, 0, 2), (5, 1, 2)]
DOOR_PASSAGE = list(set(x[0] for x in DOOR_BASE_POSITION)) + list(
    set(x[0] for x in DOOR_TARGET_POSITIONS)
)

TORCH_POSITION = (4, 0, 4)

POSSIBLE_BLOCKS = [blocks.AIR, blocks.REDSTONE_WIRE] + SOLID_BLOCKS

PISTON_GROUP = {
    blocks.STICKY_PISTON_N,
    blocks.STICKY_PISTON_S,
    blocks.STICKY_PISTON_W,
    blocks.STICKY_PISTON_E,
}


def get_global_position(x: int, y: int, z: int):
    return (
        BASE_POSITION[0] + OFFSET[0] + x,
        BASE_POSITION[1] + OFFSET[1] + y,
        BASE_POSITION[2] + OFFSET[2] + z,
    )


def validate(server_context):
    time.sleep(0.25)

    closed_blocks = 0
    for tx, ty, tz in DOOR_BASE_POSITION:
        gx, gy, gz = get_global_position(tx, ty, tz)
        response = server_context.run_command(f"execute if block {gx} {gy} {gz} minecraft:stone")
        if response == "Test passed":
            closed_blocks += 1

    server_context.set_block(*get_global_position(*TORCH_POSITION), "minecraft:redstone_torch")

    time.sleep(0.5)

    open_blocks = 0
    for tx, ty, tz in DOOR_TARGET_POSITIONS:
        gx, gy, gz = get_global_position(tx, ty, tz)
        response = server_context.run_command(f"execute if block {gx} {gy} {gz} minecraft:stone")
        if response == "Test passed":
            open_blocks += 1

    return closed_blocks + open_blocks


def evaluate(individual, server_context):
    bx, by, bz = BASE_POSITION

    server_context.clear_area(bx, by, bz, bx + 15, by + 4, bz + 15)

    for x, y, z in DOOR_BASE_POSITION:
        gx, gy, gz = get_global_position(x, y, z)
        server_context.set_block(gx, gy, gz, blocks.STONE)

    for y in range(individual.shape[0]):
        for z in range(individual.shape[1]):
            for x in range(individual.shape[2]):
                block = individual[y, z, x]
                if block == blocks.AIR:
                    continue
                gx, gy, gz = get_global_position(x, y, z)
                server_context.set_block(gx, gy, gz, block)

    score = validate(server_context)

    return (score,)


def create_individual():
    return np.array(
        [
            [
                [random.choice(POSSIBLE_BLOCKS) for x in range(BOUNDS["x"])]
                for z in range(BOUNDS["z"])
            ]
            for y in range(BOUNDS["y"])
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


def run(server_context):
    toolbox.register("evaluate", evaluate, server_context=server_context)

    POP_SIZE = 20
    NGEN = 20

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=stats, halloffame=hof, verbose=True
    )

    best = hof[0]

    evaluate(best, server_context)

    return best
