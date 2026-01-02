import logging
import time

import coloredlogs
import mcschematic
import numpy as np
from deap import algorithms, base, creator, tools
from mcschematic import MCSchematic

from . import blocks
from .blocks import SOLID_BLOCKS, Block
from .server import MinecraftServerContext

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)


# NOTE: all positions are in the Minecraft coordinate system: (x, y, z)
#        ^ Z
#        |
# +------|------+
# | . . .|. . . |
# | . . .|. . . |
# |------.------> X
# | . . .|. . . |
# | . . .|. . . |
# +------|------+
#        |
#        v

# NOTE: this defines the size of the individual within every chunk
# . . . . . . .
# . +-------+ .
# . | . . . | .
# . | . . . | .
# . | . . . | .
# . +-------+ .
# . . . . . . .
BOUNDS = np.array([5, 3, 5])


# NOTE: this is the relative offset (where the individual will take place) from the top-left block for every chunk
# . . . . ...
# . . . . ...
# . . X . ...
# . . . . ...
# . . . . ...
# : : : :  .
GLOBAL_OFFSET = np.array([4, 0, 4])


POSSIBLE_BLOCKS = [blocks.AIR, blocks.REDSTONE_DUST] + SOLID_BLOCKS


def global_position(local_position: np.ndarray, local_offset: np.ndarray):
    return local_position + GLOBAL_OFFSET + local_offset


toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


def random_torch_position():
    sx, sz = BOUNDS[[0, 2]]

    tx_min, tx_max = -1, sx
    tz_min, tz_max = -1, sz

    sidex_length = (tx_max - 1) - (tx_min + 1) + 1
    sidez_length = (tz_max - 1) - (tz_min + 1) + 1

    total_slots = 2 * sidex_length + 2 * sidez_length

    idx = np.random.randint(0, total_slots)

    if idx < sidex_length:
        tx, tz = tx_min + 1 + idx, tz_min
    elif idx < 2 * sidex_length:
        tx, tz = (tx_min + 1) + (idx - sidex_length), tz_max
    elif idx < 2 * sidex_length + sidez_length:
        tx, tz = tx_min, (tz_min + 1) + (idx - 2 * sidex_length)
    else:
        tx, tz = tx_max, (tz_min + 1) + (idx - (2 * sidex_length + sidez_length))

    return np.array([tx, -60, tz])


def create_individual():
    sx, sy, sz = BOUNDS

    individual = np.random.choice(POSSIBLE_BLOCKS, size=(sy, sz, sx))
    torch_position = random_torch_position()

    return individual, torch_position


def init_numpy_individual(icls, content_func):
    genome, torch_position = content_func()
    individual = genome.view(icls)
    individual.torch_position = torch_position
    individual.fitness = creator.FitnessMax()
    return individual


toolbox.register("individual", init_numpy_individual, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def dummy_evaluate(individual, grid_width: int, server_context: MinecraftServerContext):
    grid_x = individual.id % grid_width
    grid_z = individual.id // grid_width
    chunk_offset = np.array([grid_x * 16, 0, grid_z * 16])

    sx, sz = BOUNDS[[0, 2]]

    perimeter = []
    for lx in range(-1, sx + 1):
        for lz in range(-1, sz + 1):
            if 0 <= lx < sx and 0 <= lz < sz:
                continue
            perimeter.append((lx, lz))

    torch_position = global_position(individual.torch_position, chunk_offset)

    for lx, lz in perimeter:
        block_position = global_position(np.array([lx, -60, lz]), chunk_offset)
        if np.array_equal(block_position, torch_position):
            continue

        response = server_context.run_command(
            f"execute if block {' '.join(map(str, block_position))} minecraft:stone if block {' '.join(map(str, block_position + np.array([0, 1, 0])))} minecraft:stone"
        )
        if response and "Test passed" in response:
            return (1,)

    return (0,)


def dummy_mate(ind1, ind2):
    return ind1, ind2


toolbox.register("mate", dummy_mate)


def dummy_mutate(individual, indpb):
    for index in np.ndindex(individual.shape):
        if np.random.rand() < indpb:
            individual[index] = np.random.choice(POSSIBLE_BLOCKS)

    if np.random.rand() < 0.1:
        individual.torch_position = random_torch_position()

    return (individual,)


toolbox.register("mutate", dummy_mutate, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def build_individual(individual, offset: np.ndarray, schem: MCSchematic | None = None):
    if schem is None:
        schem = mcschematic.MCSchematic()
    for y, z, x in np.ndindex(individual.shape):
        block = individual[y, x, z]
        position = global_position(np.array([x, y, z]), offset)
        schem.setBlock(tuple(position.tolist()), block.namespaced_id)
    return schem


def preprocess_population(
    grid_width: int,
    server_context: MinecraftServerContext,
    generation_tracker: dict,
    evaluate_func,
    individuals,
):
    population = list(individuals)

    # NOTE: this does not correspond directly to the individuals
    # it is just used for positioning in the grid
    for i, ind in enumerate(population):
        ind.id = i

    server_context.clear_area((0, -60, 0), (grid_width * 16 - 1, -50, grid_width * 16 - 1))

    schem = mcschematic.MCSchematic()

    for ind in population:
        grid_x = ind.id % grid_width
        grid_z = ind.id // grid_width
        offset = np.array([grid_x * 16, 0, grid_z * 16])
        build_individual(ind, offset, schem)

    schem.save(
        server_context.schematic_folder.as_posix(),
        str(generation_tracker["current"]),
        mcschematic.Version.JE_1_21_5,
    )

    server_context.build_schematic(
        f"{server_context.schematic_folder.name}/{str(generation_tracker['current'])}"
    )

    time.sleep(0.25)

    for ind in population:
        grid_x = ind.id % grid_width
        grid_z = ind.id // grid_width
        offset = np.array([grid_x * 16, 0, grid_z * 16])

        tx, ty, tz = ind.torch_position
        torch_position = global_position(np.array([tx, ty, tz]), offset)
        server_context.set_block(torch_position, Block("minecraft:redstone_torch"))

    time.sleep(0.25)

    generation_tracker["current"] += 1

    return list(map(evaluate_func, population))


def run(server_context: MinecraftServerContext):
    POPULATION_SIZE = 20
    NGEN = 10
    generation_tracker = {"current": 0}

    grid_width = int(np.ceil(np.sqrt(POPULATION_SIZE)))

    population = toolbox.population(n=POPULATION_SIZE)

    def evaluate_wrapper(individual):
        return dummy_evaluate(individual, grid_width, server_context)

    toolbox.register("evaluate", evaluate_wrapper)
    toolbox.register("map", preprocess_population, grid_width, server_context, generation_tracker)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    server_context.clear_area((0, -60, 0), (grid_width * 16 - 1, -50, grid_width * 16 - 1))

    best_ind = hof[0]
    schem = build_individual(best_ind, np.array([0, 0, 0]))
    schem.save(
        server_context.schematic_folder.as_posix(), "best_individual", mcschematic.Version.JE_1_21_5
    )
    server_context.build_schematic(f"{server_context.schematic_folder.name}/best_individual")
    time.sleep(0.5)
    tx, ty, tz = best_ind.torch_position
    torch_position = global_position(np.array([tx, ty, tz]), np.array([0, 0, 0]))
    server_context.set_block(torch_position, Block("minecraft:redstone_torch"))
