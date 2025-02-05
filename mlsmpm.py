import argparse
import logging

import taichi as ti

from opt import TaichiArch

logger = logging.getLogger(__name__)


def main():
    pass


if __name__ == "__main__":
    logging.basicConfig(encoding="utf-8", level=logging.INFO)

    parser = argparse.ArgumentParser(prog="MLS-MPM Incremental")

    parser.add_argument(
        "--arch",
        type=TaichiArch,
        choices=list(TaichiArch),
        help="Taichi Arch",
        default=TaichiArch.cpu,
    )

    opts = parser.parse_args()

    logger.info(f"Taichi Arch: {opts.arch}")
    ti.init(arch=opts.arch.decode())
    main()
