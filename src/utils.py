import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mcschematic
import numpy as np

from .blocks import Block
from .pipeline import build_individual, global_position
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

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    cmap = plt.cm.get_cmap("nipy_spectral")
    num_files = len(pickle_files)
    colours = [cmap(i) for i in np.linspace(0, 0.85, num_files)]

    lines = []
    labels = []

    for i, (f, c) in enumerate(zip(pickle_files, colours)):
        with open(f, "rb") as pf:
            data = pickle.load(pf)

        logbook = data["logbook"]

        generations = np.array(logbook.select("gen"))

        chapter_fitness = logbook.chapters["fitness"]
        fitness_avgs = np.array(chapter_fitness.select("avg"))
        fitness_stds = np.array(chapter_fitness.select("std"))
        fitness_maxs = np.array(chapter_fitness.select("max"))

        chapter_blocks = logbook.chapters["blocks"]
        blocks_avgs = np.array(chapter_blocks.select("avg"))
        blocks_mins = np.array(chapter_blocks.select("min"))

        label = f"Run {i+1}"

        (l1,) = ax1.plot(
            generations,
            fitness_maxs,
            color=c,
            linestyle="-",
            linewidth=2,
            label=f"{label} fitness max",
        )
        (l2,) = ax1.plot(
            generations,
            fitness_avgs,
            color=c,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"{label} fitness avg",
        )
        ax1.fill_between(
            generations,
            fitness_avgs - fitness_stds,
            fitness_avgs + fitness_stds,
            color=c,
            alpha=0.1,
        )

        (l3,) = ax2.plot(
            generations,
            blocks_mins,
            color=c,
            linestyle=":",
            linewidth=2,
            alpha=0.8,
            label=f"{label} blocks min",
        )
        (l4,) = ax2.plot(
            generations,
            blocks_avgs,
            color=c,
            linestyle="-.",
            linewidth=1,
            alpha=0.5,
            label=f"{label} blocks avg",
        )

        lines.extend([l1, l2, l3, l4])
        labels.extend([l1.get_label(), l2.get_label(), l3.get_label(), l4.get_label()])

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax2.set_ylabel("Block Count")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.15, 1), loc="upper left")

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
