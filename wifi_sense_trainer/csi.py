"""CSI utilities: amplitude fusion, normalization, and a /ws/sensing client.

The aggregator (RuView ruvnet/wifi-densepose) broadcasts `sensing_update` frames:
  {"type":"sensing_update","timestamp":...,"nodes":[{"node_id","amplitude":[...],...}], ...}
This module turns the per-node amplitude into one fixed-width vector and offers a
small reconnecting WebSocket client so teacher/infer share one code path.
"""
import json
import os
import time
import numpy as np

N_SUB = 56  # RuView MM-Fi subcarrier width


def fuse_amplitude(nodes, n_sub=N_SUB, mode="mean"):
    """Per-node CSI amplitude -> single (n_sub,) float32 vector, or None."""
    amps = []
    for n in nodes or []:
        a = n.get("amplitude")
        if not a:
            continue
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            continue
        if a.size != n_sub:  # resample any width -> n_sub
            a = np.interp(np.linspace(0, 1, n_sub), np.linspace(0, 1, a.size), a).astype(np.float32)
        amps.append(a)
    if not amps:
        return None
    stacked = np.vstack(amps)
    return (stacked[0] if mode == "node0" else stacked.mean(0)).astype(np.float32)


class Normalizer:
    """Per-feature z-score; persisted alongside the model so infer replays it."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32) + 1e-6

    @classmethod
    def fit(cls, X):
        return cls(X.mean(0), X.std(0))

    def apply(self, x):
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def to_json(self, path, extra=None):
        d = {"mean": self.mean.tolist(), "std": (self.std - 1e-6).tolist()}
        if extra:
            d.update(extra)
        json.dump(d, open(path, "w"))

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls(d["mean"], d["std"])


class SensingWS:
    """Reconnecting client for the aggregator's /ws/sensing. Calls on_frame(dict)
    for every sensing_update message. Blocking; run in a thread if needed."""

    def __init__(self, url, on_frame):
        self.url = url
        self.on_frame = on_frame
        self._stop = False

    def _handle(self, _ws, msg):
        try:
            d = json.loads(msg)
        except Exception:
            return
        if d.get("type") == "sensing_update":
            self.on_frame(d)

    def run_forever(self):
        import websocket  # websocket-client
        while not self._stop:
            try:
                websocket.WebSocketApp(self.url, on_message=self._handle).run_forever(
                    ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            if not self._stop:
                time.sleep(2)

    def stop(self):
        self._stop = True
