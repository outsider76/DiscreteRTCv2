#!/usr/bin/env python3
"""Asynchronous Piper control with timestamp-aligned temporal ensembling.

Inference runs in a background thread while the 50 Hz controller continues.
For a controller timestamp covered by both the previous and newest chunks,
the two predicted actions are combined with a configurable weighted average.
Dry-run is the default.
"""

from examples.realRobots.Piper.eval_files.piper_async_common import (
    build_async_parser,
    run_async_controller,
)


def main() -> None:
    parser = build_async_parser("temporal_ensemble", __doc__)
    run_async_controller("temporal_ensemble", parser.parse_args())


if __name__ == "__main__":
    main()
