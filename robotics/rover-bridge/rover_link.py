"""
rover_link.py — serial link to the rover's STM32 drive controller, with a small command-line test.

Usage:
    python3 rover_link.py ping
    python3 rover_link.py drive 60 60 --seconds 2
    python3 rover_link.py stop
    python3 rover_link.py raw "C LIMIT 100"
"""

# %% Imports
import argparse
import time

import serial


# %% Config
PORT          = "/dev/ttyTHS1"
BAUD          = 115200
READ_TIMEOUT  = 0.15     # seconds to wait for one reply line
SPEED_MAX     = 255
SEND_PERIOD   = 0.1      # seconds between V commands, well inside the 500 ms failsafe


# %% Link
class RoverLink:
    """Text-line protocol over a UART.

    The Jetson's header UART zeroes replies of 16 bytes or more, so only rely on the short
    replies (OK, ERR, PONG, FAILSAFE).
    """

    def __init__(self, port: str = PORT, baud: int = BAUD) -> None:
        self.ser = serial.Serial(port, baud, timeout=READ_TIMEOUT)
        time.sleep(0.3)
        self.ser.write(b"\n")                # the first byte after opening the port is lost
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def send(self, command: str) -> str:
        """Send one line and return the first reply that is not a FAILSAFE notice."""
        self.ser.write(command.encode("ascii") + b"\n")

        for _ in range(3):
            line = self.ser.readline().decode("ascii", errors="replace").strip()
            if line != "FAILSAFE":
                return line
        return ""

    def ping(self) -> bool:
        return self.send("P") == "PONG"

    def drive(self, left: int, right: int) -> bool:
        left  = max(-SPEED_MAX, min(SPEED_MAX, int(left)))
        right = max(-SPEED_MAX, min(SPEED_MAX, int(right)))
        return self.send(f"V {left} {right}") == "OK"

    def stop(self) -> bool:
        return self.send("S") == "OK"

    def configure(self, settings: dict) -> bool:
        """Send C settings such as {"limit": 100, "pwm": 16000}; True if every one was accepted."""
        ok = True
        for key, value in settings.items():
            ok = (self.send(f"C {key.upper()} {value}") == "OK") and ok
        return ok

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.ser.close()

    def __enter__(self) -> "RoverLink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# %% Command line
def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to the rover drive controller")
    parser.add_argument("--port", default=PORT)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ping")
    sub.add_parser("stop")
    drive_p = sub.add_parser("drive")
    drive_p.add_argument("left",  type=int)
    drive_p.add_argument("right", type=int)
    drive_p.add_argument("--seconds", type=float, default=2.0)
    raw_p = sub.add_parser("raw")
    raw_p.add_argument("line")

    args = parser.parse_args()

    with RoverLink(args.port) as rover:
        if args.command == "ping":
            print("PONG" if rover.ping() else "no reply")
        elif args.command == "stop":
            print("stopped" if rover.stop() else "no reply")
        elif args.command == "raw":
            print(repr(rover.send(args.line)))
        elif args.command == "drive":
            end, sent, ok = time.time() + args.seconds, 0, 0
            while time.time() < end:
                ok   += rover.drive(args.left, args.right)
                sent += 1
                time.sleep(SEND_PERIOD)
            print(f"{ok}/{sent} commands acknowledged")


# %% Entry point
if __name__ == "__main__":
    main()
