"""
Receives 84x84 JPEG frames from rs_udp_sender.py and saves them to disk.

Usage:
  python3 test_udp_image.py [--port PORT] [--save-dir DIR] [--max-frames N]
"""

import argparse
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np

BIND_PORT = 5010
SAVE_DIR = "./received_frames"
SAVE_HZ = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=BIND_PORT)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--save-hz", type=float, default=SAVE_HZ,
                        help="Save rate in Hz (default 2)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N saved frames (0 = run until Ctrl+C)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    print(f"Listening on port {args.port}. Saving at {args.save_hz} Hz to '{save_dir}/'")
    print("Press Ctrl+C to stop.\n")

    save_interval = 1.0 / args.save_hz
    last_save_time = 0.0
    count = 0

    try:
        while True:
            data, addr = sock.recvfrom(70000)

            if len(data) < 8:
                print(f"[WARN] Packet too short ({len(data)}B), skipping")
                continue

            seq, jpeg_len = struct.unpack_from(">II", data, 0)
            jpeg_bytes = data[8: 8 + jpeg_len]

            if len(jpeg_bytes) != jpeg_len:
                print(f"[WARN] seq={seq} truncated ({len(jpeg_bytes)}/{jpeg_len}B), skipping")
                continue

            now = time.monotonic()
            if now - last_save_time < save_interval:
                continue
            last_save_time = now

            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] seq={seq} JPEG decode failed, skipping")
                continue

            path = save_dir / f"frame_{seq:06d}.jpg"
            cv2.imwrite(str(path), bgr)
            count += 1
            print(f"seq={seq:06d}  from={addr[0]}:{addr[1]}  size={bgr.shape[1]}x{bgr.shape[0]}  saved → {path.name}")

            if args.max_frames > 0 and count >= args.max_frames:
                print(f"\nReached {args.max_frames} frames. Done.")
                break

    except KeyboardInterrupt:
        print(f"\nStopped. Saved {count} frames to '{save_dir}/'.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
