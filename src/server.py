import logging
import re
import subprocess
import time
from pathlib import Path

import coloredlogs
from mcrcon import MCRcon

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


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
            logger.info("Minecraft server process terminated.")
