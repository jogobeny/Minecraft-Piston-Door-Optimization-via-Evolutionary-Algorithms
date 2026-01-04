import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mcschematic
import numpy as np

from .blocks import Block
from .pipeline import POSSIBLE_BLOCKS, build_individual, global_position
from .server import MinecraftServer


def build(args):
    with open(args.pickle_file, "rb") as f:
        data = pickle.load(f)

    hof = data["hof"]
    best_individual = hof[0]

    with MinecraftServer(
        ip_address=args.ip,
        rcon_port=args.rcon_port,
        rcon_password=args.rcon_password,
        server_jar_path=args.server_jar_path,
        java_path=args.java_path,
    ).as_context() as ctx:
        ctx.clear_area((0, -60, 0), (1 * 16 - 1, -60 + 5 * 2, 1 * 16 - 1))  # TODO: harcoded y=5

        schem = build_individual(best_individual, np.array([0, 0, 0]))
        schem.save(
            ctx.schematic_folder.as_posix(),
            "best_individual",
            mcschematic.Version.JE_1_21_5,
        )
        ctx.build_schematic(f"{ctx.schematic_folder.name}/best_individual")

        time.sleep(0.25)

        tx, ty, tz = best_individual.torch_position
        torch_position = global_position(
            np.array([tx, ty, tz]), np.array([0, -60 if ty == 0 else 0, 0])
        )
        ctx.set_block(torch_position, Block("minecraft:redstone_torch"))


def graph(args):
    folder_path = Path(args.folder)
    pickle_files = list(folder_path.glob("*.pkl"))

    with open(pickle_files[0], "rb") as pf:
        test_data = pickle.load(pf)
        has_weights = "weights" in test_data["logbook"].chapters

    n_rows = 3 if has_weights else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows + 2), sharex=True)

    if n_rows == 1:
        axes = [axes]
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_weights else None

    cmap = plt.get_cmap("nipy_spectral")
    colours = [cmap(i) for i in np.linspace(0, 0.85, len(pickle_files))]

    weights_colors = [plt.cm.tab20(i % 20) for i in range(len(POSSIBLE_BLOCKS))]
    weights_labels = [b.namespaced_id.split(":")[-1] for b in POSSIBLE_BLOCKS]
    weights_generations = None
    weights_data = None

    for i, (f, c) in enumerate(zip(pickle_files, colours)):
        with open(f, "rb") as pf:
            data = pickle.load(pf)

        label = f"Run {i+1}"
        logbook = data["logbook"]
        generations = np.array(logbook.select("gen"))

        chapter_fitness = logbook.chapters["fitness"]
        fitness_avgs = np.array(chapter_fitness.select("avg"))
        fitness_stds = np.array(chapter_fitness.select("std"))
        fitness_maxs = np.array(chapter_fitness.select("max"))

        if fitness_avgs.ndim > 1:
            fitness_avgs = fitness_avgs[:, 0]
            fitness_stds = fitness_stds[:, 0]
            fitness_maxs = fitness_maxs[:, 0]

        ax1.plot(generations, fitness_maxs, color=c, ls="-", lw=2, label=f"{label} - max")
        ax1.plot(
            generations, fitness_avgs, color=c, ls="--", lw=1, alpha=0.7, label=f"{label} - avg"
        )
        ax1.fill_between(
            generations,
            fitness_avgs - fitness_stds,
            fitness_avgs + fitness_stds,
            color=c,
            alpha=0.1,
        )

        if "blocks" in logbook.chapters:
            chapter_blocks = logbook.chapters["blocks"]
            blocks_avgs = np.array(chapter_blocks.select("avg"))
            blocks_mins = np.array(chapter_blocks.select("min"))

            ax2.plot(generations, blocks_mins, color=c, ls=":", lw=2, label=f"{label} - min")
            ax2.plot(
                generations, blocks_avgs, color=c, ls="-.", lw=1, alpha=0.7, label=f"{label} - avg"
            )

        if has_weights and "weights" in logbook.chapters:
            chapter_weights = logbook.chapters["weights"]
            weights_avgs = np.array(chapter_weights.select("avg"))
            weights_generations = generations
            weights_data = weights_avgs.T

    ax1.set_ylabel("Fitness")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")

    ax2.set_ylabel("Block Count")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")

    if weights_generations is not None and weights_data is not None:
        ax3.stackplot(
            weights_generations,
            weights_data,
            labels=weights_labels,
            colors=weights_colors,
            alpha=0.8,
        )
        ax3.set_ylabel("Block Probability")
        ax3.set_xlabel("Generation")
        ax3.set_ylim(0, 1.0)
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True, help="Choose a command")

    parser_build = subparsers.add_parser("build", help="Build the best individual")
    parser_build.add_argument("pickle_file", type=str, help="Path to the pickle file of the run")
    parser_build.add_argument(
        "--ip",
        help="IP address of the RCON server. If provided, the server will NOT be started locally.",
    )
    parser_build.add_argument("--rcon-port", type=int, default=25575, help="RCON port")
    parser_build.add_argument("--rcon-password", default="1234", help="RCON password")
    parser_build.add_argument(
        "--server-jar-path",
        type=Path,
        default=Path(__file__).parent.parent / "server" / "server.jar",
        help="Minecraft server JAR file path",
    )
    parser_build.add_argument(
        "--java-path",
        default="/usr/lib/jvm/java-21-openjdk/bin/java",
        help="Path to Java 21 executable",
    )
    parser_build.set_defaults(func=build)

    parser_graph = subparsers.add_parser("graph", help="Graph the results of a run")
    parser_graph.add_argument("folder", type=str, help="Path to the folder of the run")
    parser_graph.set_defaults(func=graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
