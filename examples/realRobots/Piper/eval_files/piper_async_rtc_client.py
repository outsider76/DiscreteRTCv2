#!/usr/bin/env python3
"""Asynchronous Piper control with inference-time Real-Time Chunking.

The 50 Hz control loop continues consuming the current action chunk while a
background thread requests the next chunk.  The RTC server conditions the new
flow-matching sample on the timestamp-aligned, unexecuted prefix of the current
chunk using ΠGDM soft-mask guidance.  Dry-run is the default.
"""

from examples.realRobots.Piper.eval_files.piper_async_common import (
    build_async_parser,
    run_async_controller,
)


def main() -> None:
    parser = build_async_parser("rtc", __doc__)
    run_async_controller("rtc", parser.parse_args())


if __name__ == "__main__":
    main()
