import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import coloredlogs
import numpy as np
from mcrcon import MCRcon

from .blocks import Block

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)


class MinecraftServer:

    def __init__(
        self,
        *,
        ip_address: str | None,
        rcon_port: int,
        rcon_password: str,
        server_jar_path: Path,
        java_path: str,
    ):
        self.ip_address = ip_address
        self.rcon_port = rcon_port
        self.rcon_password = rcon_password
        self.server_jar_path = server_jar_path
        self.java_path = java_path

    def as_context(self):
        return MinecraftServerContext(self)

    def start_server(self, *, params: list[str] = ["-Xmx2G", "-Xms2G"], timeout: int = 60):
        if self.ip_address is not None:
            return

        cmd = [
            self.java_path,
            *params,
            "-jar",
            str(self.server_jar_path),
            "nogui",
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.server_jar_path.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logger.info("Starting Minecraft server...")

        done_regex = re.compile(r"Done \(\d+\.\d+s\)!")
        rcon_regex = re.compile(r"RCON running on")

        done_seen = False
        rcon_seen = False

        start_time = time.time()

        for line in self.process.stdout:
            line = line.rstrip()

            if done_regex.search(line):
                done_seen = True
                logger.debug("Server startup DONE detected.")

            if rcon_regex.search(line):
                rcon_seen = True
                logger.debug("RCON startup detected.")

            if done_seen and rcon_seen:
                logger.info("Minecraft server started.")
                return

            if time.time() - start_time > timeout:
                raise TimeoutError("Minecraft server did not start in time.")

        raise RuntimeError("Minecraft server process ended unexpectedly.")

    def connect_to_server(self):
        ip = self.ip_address or "127.0.0.1"
        self.rcon = MCRcon(ip, self.rcon_password, self.rcon_port)
        self.rcon.connect()
        logger.info(f"Successfully connected to RCON at {ip}:{self.rcon_port}")

    def kill_server(self):
        logger.info("Stopping Minecraft server...")

        if hasattr(self, "rcon"):
            self.rcon.command("stop")
            self.rcon.disconnect()
            logger.info("RCON disconnected.")

        if hasattr(self, "process"):
            self.process.terminate()
            self.process.wait()
            logger.info("Minecraft server stopped.")


class MinecraftServerContext:

    def __init__(self, server: MinecraftServer):
        self.server = server
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.schematic_folder = (
            self.server.server_jar_path.parent
            / "plugins"
            / "FastAsyncWorldEdit"
            / "schematics"
            / self.start_datetime
        )
        self.schematic_folder.mkdir(parents=True, exist_ok=True)

        # NOTE: this is the absolute base position of the grid (the top-left block)
        # X . . ...
        # . . . ...
        # . . . ...
        # : : :  .
        self.BASE_POSITION = np.array([0, -60, 0])

    def __enter__(self):
        self.server.start_server()
        self.server.connect_to_server()
        self.run_command("//world world")  # You need to provide a world (Try //world)
        self.run_command(
            f"//pos1 {','.join(map(str, self.BASE_POSITION))}"
        )  # Make a region selection first.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.server.ip_address is None:
            self.server.kill_server()

        return exc_type is KeyboardInterrupt

    def run_command(self, cmd: str) -> str:
        response = self.server.rcon.command(cmd)
        logger.debug(f"Command: {cmd}, Response: {response}")
        return response

    def set_block(self, position: tuple[int, int, int], block: Block):
        x, y, z = position
        self.run_command(f"setblock {x} {y} {z} {block.namespaced_id}")

    def clear_area(self, position1: tuple[int, int, int], position2: tuple[int, int, int]):
        self.run_command(f"//pos1 {','.join(map(str, position1))}")
        self.run_command(f"//pos2 {','.join(map(str, position2))}")
        self.run_command("//set minecraft:air")

        x1, y1, z1 = position1
        x2, y2, z2 = position2
        self.run_command(
            f"kill @e[type=item, x={x1}, y={y1}, z={z1}, dx={x2-x1}, dy={y2-y1}, dz={z2-z1}]"
        )

    def build_schematic(self, schematic_filepath: str):
        self.run_command(f"/schematic load {schematic_filepath}")
        self.run_command("//paste -a")
