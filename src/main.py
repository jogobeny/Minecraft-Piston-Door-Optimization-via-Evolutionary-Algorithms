import argparse
import pickle
from argparse import Namespace
from pathlib import Path

from .pipeline import run
from .server import MinecraftServer


class Args(Namespace):
    ip: str | None
    rcon_port: int
    rcon_password: str
    server_jar_path: Path
    java_path: str


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ip",
        help="IP address of the RCON server. If provided, the server will NOT be started locally.",
    )
    parser.add_argument("--rcon-port", type=int, default=25575, help="RCON port")
    parser.add_argument("--rcon-password", default="1234", help="RCON password")
    parser.add_argument(
        "--server-jar-path",
        type=Path,
        default=Path(__file__).parent.parent / "server" / "server.jar",
        help="Minecraft server JAR file path",
    )
    parser.add_argument(
        "--java-path",
        default="/usr/lib/jvm/java-21-openjdk/bin/java",
        help="Path to Java 21 executable",
    )

    args = parser.parse_args(namespace=Args())

    with MinecraftServer(
        ip_address=args.ip,
        rcon_port=args.rcon_port,
        rcon_password=args.rcon_password,
        server_jar_path=args.server_jar_path,
        java_path=args.java_path,
    ).as_context() as ctx:
        population, logbook, hof = run(ctx)

        with open(f"{ctx.start_datetime}.pkl", "wb") as f:
            pickle.dump({"population": population, "logbook": logbook, "hof": hof}, f)


if __name__ == "__main__":
    main()
