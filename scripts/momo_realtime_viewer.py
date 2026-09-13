"""Plot one observer from UDP, or replay the same JSONL schema offline."""

import argparse
from collections import deque
import json
import socket

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


FIELDS = [
    ("local_residual", slice(0, 3), "Pelvis-frame base residual [N]"),
    ("local_residual", slice(3, 6), "Pelvis-frame base torque [Nm]"),
    ("local_residual", slice(6, 35), "Joint residuals [Nm]"),
    ("scaled_residual", slice(0, 35), "Effort-scaled residual"),
    ("actor_residual", slice(None), "Actual actor residual (light-step = zero)"),
    ("linear_velocity", slice(None), "Pelvis velocity [m/s]"),
]


def draw(axes, records):
    if not records:
        return
    times = np.array([r["time"] for r in records])
    for ax, (field, indices, title) in zip(axes, FIELDS):
        ax.clear()
        values = np.array([r[field] for r in records])[:, indices]
        ax.plot(times, values, linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", help="Replay JSONL instead of receiving UDP")
    parser.add_argument("--port", type=int, default=9870)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--output", help="Save a static figure from --log (headless supported)")
    args = parser.parse_args()
    if args.output and not args.log:
        parser.error("--output requires --log")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()
    records = deque(maxlen=args.window)
    if args.log:
        with open(args.log) as stream:
            for line in stream:
                record = json.loads(line)
                if record["schema"] != 1:
                    raise ValueError("Unsupported telemetry schema")
                records.append(record)
        draw(axes, records)
        if args.output:
            fig.savefig(args.output, dpi=150)
        else:
            plt.show()
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((args.bind, args.port))
        sock.setblocking(False)

        def update(_):
            while True:
                try:
                    record = json.loads(sock.recv(65535))
                    if record["schema"] == 1:
                        records.append(record)
                except BlockingIOError:
                    break
            draw(axes, records)

        animation = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
        try:
            plt.show()
        finally:
            sock.close()
        del animation


if __name__ == "__main__":
    main()
