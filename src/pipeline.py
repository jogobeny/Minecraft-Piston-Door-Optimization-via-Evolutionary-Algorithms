import logging
import time
from pathlib import Path

import coloredlogs
import joblib
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

    individual = np.empty((sy, sz, sx), dtype=object)
    for y in range(sy):
        for z in range(sz):
            for x in range(sx):
                not_redstone = False
                if y > 0:
                    block_below = individual[y - 1, z, x]
                    if block_below not in SOLID_BLOCKS:
                        not_redstone = True

                if not_redstone:
                    individual[y, z, x] = np.random.choice(SOLID_BLOCKS + [blocks.AIR])
                else:
                    individual[y, z, x] = np.random.choice(POSSIBLE_BLOCKS)

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


def dummy_evaluate(
    individual, grid_width: int, schematic_on: MCSchematic, schematic_off: MCSchematic
):
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

    max_score = 0

    def get_block_from_schematic(position: np.ndarray, schematic: MCSchematic):
        schematic_position = (
            position[0],
            position[1] + 60,
            position[2],
        )  # schematic is relative to y=-60

        block_id = schematic._structure._blockStates.get(schematic_position)
        if block_id is None:
            return blocks.AIR.namespaced_id

        return schematic._structure._blockPalette[block_id]

    for lx, lz in perimeter:
        block_position = global_position(np.array([lx, -60, lz]), chunk_offset)
        if np.array_equal(block_position, torch_position):
            continue

        # NOTE: strict evaluation
        # response = server_context.run_command(
        #     f"execute if block {' '.join(map(str, block_position))} minecraft:stone if block {' '.join(map(str, block_position + np.array([0, 1, 0])))} minecraft:stone"
        # )
        # if response and "Test passed" in response:
        #     return (1,)

        score_lower = 0
        block_on = get_block_from_schematic(block_position, schematic_on)
        block_off = get_block_from_schematic(block_position, schematic_off)
        if "minecraft:stone" in block_on:
            score_lower += 0.5
            if "minecraft:air" in block_off:
                score_lower += 0.5
        elif "minecraft:air" not in block_on:
            score_lower += 0.1

        score_upper = 0
        block_on = get_block_from_schematic(block_position + np.array([0, 1, 0]), schematic_on)
        block_off = get_block_from_schematic(block_position + np.array([0, 1, 0]), schematic_off)
        if "minecraft:stone" in block_on:
            score_upper += 0.5
            if "minecraft:air" in block_off:
                score_upper += 0.5
        elif "minecraft:air" not in block_on:
            score_upper += 0.1

        max_score = max(max_score, max(score_lower, score_upper))

        if max_score >= 1.0:
            return (1.0,)

    return (max_score,)


def dummy_mate(ind1, ind2):
    axis = np.random.choice([1, 2])  # 1: z-axis, 2: x-axis

    point = np.random.randint(1, ind1.shape[axis])

    if axis == 1:
        ind1[:, point:, :], ind2[:, point:, :] = (
            ind2[:, point:, :].copy(),
            ind1[:, point:, :].copy(),
        )
    else:
        ind1[:, :, point:], ind2[:, :, point:] = (
            ind2[:, :, point:].copy(),
            ind1[:, :, point:].copy(),
        )

    if np.random.rand() < 0.1:
        ind1.torch_position, ind2.torch_position = (
            ind2.torch_position.copy(),
            ind1.torch_position.copy(),
        )

    return ind1, ind2


toolbox.register("mate", dummy_mate)


def dummy_mutate(individual, indpb):
    sy, sz, sx = individual.shape

    for y in range(sy):
        for z in range(sz):
            for x in range(sx):
                if np.random.rand() < indpb:
                    not_redstone = False
                    if y > 0:
                        block_below = individual[y - 1, z, x]
                        if block_below not in SOLID_BLOCKS:
                            not_redstone = True

                    if not_redstone:
                        individual[y, z, x] = np.random.choice(SOLID_BLOCKS + [blocks.AIR])
                    else:
                        individual[y, z, x] = np.random.choice(POSSIBLE_BLOCKS)

                    if y < sy - 1:
                        block_above = individual[y + 1, z, x]
                        if (
                            block_above == blocks.REDSTONE_DUST
                            and individual[y, z, x] not in SOLID_BLOCKS
                        ):
                            individual[y + 1, z, x] = blocks.AIR

    if np.random.rand() < 0.1:
        individual.torch_position = random_torch_position()

    return (individual,)


toolbox.register("mutate", dummy_mutate, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def build_individual(individual, offset: np.ndarray, schem: MCSchematic | None = None):
    if schem is None:
        schem = mcschematic.MCSchematic()

    for y, z, x in np.ndindex(individual.shape):
        block = individual[y, z, x]
        position = global_position(np.array([x, y, z]), offset)
        schem.setBlock(tuple(position.tolist()), block.namespaced_id)
    return schem


def wait_for_file(filepath: Path, timeout: float = 5.0):
    start_time = time.monotonic()

    # TODO: i guess this could be problematic if the file is created but not yet fully written
    while not filepath.is_file():
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for file: {filepath}")
        time.sleep(0.05)


def preprocess_population(
    grid_width: int,
    server_context: MinecraftServerContext,
    generation_tracker: dict,
    evaluate_func,
    individuals,
):
    population = list(individuals)

    # NOTE: bulding and evaluating individuals are expensive
    # this will hash (genome, torch_position) to avoid duplicates
    # `toolbox.map` expects fitnesses as return value, so we can do it here
    population_with_hashes = []
    for ind in population:
        signature = (ind.view(np.ndarray), ind.torch_position)
        population_with_hashes.append((joblib.hash(signature), ind))

    unique_list = []
    unique_hashes = []
    seen_hashes = set()

    for h, ind in population_with_hashes:
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_hashes.append(h)
            unique_list.append(ind)

    # NOTE: this does not correspond directly to the individuals
    # it is just used for positioning in the grid
    for i, ind in enumerate(unique_list):
        ind.id = i

    server_context.clear_area((0, -60, 0), (grid_width * 16 - 1, -50, grid_width * 16 - 1))

    schem = mcschematic.MCSchematic()

    for ind in unique_list:
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

    # == phase 1: with redstone torches ==

    for ind in unique_list:
        grid_x = ind.id % grid_width
        grid_z = ind.id // grid_width
        offset = np.array([grid_x * 16, 0, grid_z * 16])

        tx, ty, tz = ind.torch_position
        torch_position = global_position(np.array([tx, ty, tz]), offset)
        server_context.set_block(torch_position, Block("minecraft:redstone_torch"))

    time.sleep(0.5)

    server_context.run_command(f"//pos1 0,-60,0")
    server_context.run_command(f"//pos2 {grid_width * 16 - 1},-56,{grid_width * 16 - 1}")
    server_context.run_command("//copy")
    server_context.run_command(
        f"//schematic save {server_context.schematic_folder.name}/{str(generation_tracker['current'])}.eval_on sponge.2"
    )

    wait_for_file(
        server_context.schematic_folder / f"{str(generation_tracker['current'])}.eval_on.schem"
    )

    schem_on = mcschematic.MCSchematic(
        f"{server_context.schematic_folder / str(generation_tracker['current'])}.eval_on.schem"
    )

    # == phase 2: without redstone torches ==

    for ind in unique_list:
        grid_x = ind.id % grid_width
        grid_z = ind.id // grid_width
        offset = np.array([grid_x * 16, 0, grid_z * 16])

        tx, ty, tz = ind.torch_position
        torch_position = global_position(np.array([tx, ty, tz]), offset)
        server_context.set_block(torch_position, blocks.AIR)

    time.sleep(0.5)

    server_context.run_command(f"//pos1 0,-60,0")
    server_context.run_command(f"//pos2 {grid_width * 16 - 1},-56,{grid_width * 16 - 1}")
    server_context.run_command("//copy")
    server_context.run_command(
        f"//schematic save {server_context.schematic_folder.name}/{str(generation_tracker['current'])}.eval_off sponge.2"
    )

    wait_for_file(
        server_context.schematic_folder / f"{str(generation_tracker['current'])}.eval_off.schem"
    )

    schem_off = mcschematic.MCSchematic(
        f"{server_context.schematic_folder / str(generation_tracker['current'])}.eval_off.schem"
    )

    # == phase 3: evaluate ==

    fitness_cache = {}
    for ind, h in zip(unique_list, unique_hashes):
        fitness = dummy_evaluate(ind, grid_width, schem_on, schem_off)
        fitness_cache[h] = fitness

    fitnesses = []
    for h, ind in population_with_hashes:
        fitness = fitness_cache[h]
        ind.fitness.values = fitness
        fitnesses.append(fitness)

    generation_tracker["current"] += 1

    return fitnesses


def run(server_context: MinecraftServerContext, build_best_at_end: bool = True):
    POPULATION_SIZE = 50
    NGEN = 100

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
    stats.register("std", np.std)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    if build_best_at_end:
        server_context.clear_area((0, -60, 0), (grid_width * 16 - 1, -50, grid_width * 16 - 1))

        best_ind = hof[0]
        schem = build_individual(best_ind, np.array([0, 0, 0]))
        schem.save(
            server_context.schematic_folder.as_posix(),
            "best_individual",
            mcschematic.Version.JE_1_21_5,
        )
        server_context.build_schematic(f"{server_context.schematic_folder.name}/best_individual")

        time.sleep(0.25)

        tx, ty, tz = best_ind.torch_position
        torch_position = global_position(np.array([tx, ty, tz]), np.array([0, 0, 0]))
        server_context.set_block(torch_position, Block("minecraft:redstone_torch"))

    return population, logbook, hof
