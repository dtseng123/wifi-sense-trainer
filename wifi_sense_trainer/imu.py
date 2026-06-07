"""Minimal threaded reader for an ESP32-S3 sensor-hub IMU streaming over USB
serial. Self-contained (no extra repo deps) so a teacher rig that co-mounts the
stereo camera with an IMU can stamp each training sample with the rig's
orientation quaternion.

Wire protocol (35-byte binary packet, little-endian):
    header(0xAA) | timestamp_ms(uint32) | qw qx qy qz (float32 x4)
                 | ax ay az (float32 x3) | checksum(xor) | footer(0x55)
checksum = XOR of every byte between header and checksum (exclusive).

Requires pyserial (`pip install wifi-sense-trainer[imu]`). If pyserial is
missing or the port can't be opened, ImuReader.start() returns False and the
caller should simply continue without orientation tags.
"""
import struct
import threading
import time

PACKET_SIZE = 35
HEADER = 0xAA
FOOTER = 0x55
_FMT = "<Ifffffff"  # timestamp_ms, qw, qx, qy, qz, ax, ay, az


def _checksum_ok(pkt):
    c = 0
    for b in pkt[1:-2]:  # exclude header, checksum, footer
        c ^= b
    return c == pkt[-2]


class ImuReader:
    """Background serial reader. latest() returns the most recent
    (qw, qx, qy, qz, ax, ay, az) tuple, or None until the first valid packet."""

    def __init__(self, port="/dev/ttyACM0", baud=115200):
        self.port, self.baud = port, baud
        self._serial = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest = None
        self.packets = 0
        self.invalid = 0

    def start(self):
        try:
            import serial
        except ImportError:
            print("[imu] pyserial not installed; install with "
                  "`pip install wifi-sense-trainer[imu]` — continuing without IMU")
            return False
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=0.01)
            self._serial.reset_input_buffer()
        except Exception as e:  # serial.SerialException etc.
            print(f"[imu] cannot open {self.port}: {e} — continuing without IMU")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[imu] reading {self.port} @ {self.baud}")
        return True

    def _loop(self):
        buf = bytearray()
        while self._running:
            try:
                waiting = self._serial.in_waiting
            except OSError:
                print("[imu] disconnected")
                break
            if waiting:
                buf.extend(self._serial.read(waiting))
            else:
                time.sleep(0.0005)
                continue
            while len(buf) >= PACKET_SIZE:
                i = buf.find(HEADER)
                if i < 0:
                    buf.clear()
                    break
                if i > 0:
                    del buf[:i]
                if len(buf) < PACKET_SIZE:
                    break
                if buf[PACKET_SIZE - 1] != FOOTER:
                    del buf[0]
                    continue
                pkt = bytes(buf[:PACKET_SIZE])
                del buf[:PACKET_SIZE]
                if not _checksum_ok(pkt):
                    self.invalid += 1
                    continue
                try:
                    v = struct.unpack(_FMT, pkt[1:33])
                except struct.error:
                    self.invalid += 1
                    continue
                with self._lock:
                    self._latest = v[1:]  # drop timestamp -> (qw..qz, ax..az)
                self.packets += 1

    def latest(self):
        with self._lock:
            return self._latest

    def latest_quat(self):
        """Most recent (qw, qx, qy, qz), or (1,0,0,0) identity if no data yet."""
        v = self.latest()
        return tuple(v[:4]) if v else (1.0, 0.0, 0.0, 0.0)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
