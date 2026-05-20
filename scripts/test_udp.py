#!/usr/bin/env python3
import argparse
import socket
import struct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-ip", default="100.94.172.95", type=str)
    parser.add_argument("--result-port", default=5009, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_vector = (1.0, 0.0, 0.0)
    payload = struct.pack("3d", *action_vector)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, (args.result_ip, args.result_port))
    finally:
        sock.close()

    print(
        f"Sent test right_constraint {list(action_vector)} "
        f"to {args.result_ip}:{args.result_port}"
    )


if __name__ == "__main__":
    main()
