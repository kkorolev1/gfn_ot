"""Experiment logging, local artifacts, and a single replaceable checkpoint."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
from tempfile import mkdtemp

from flax import serialization
import numpy as np

from utils.helper import flatten_dict


def _json_value(value):
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if key in {"api_key", "password", "access_token"}
            else _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _json_value(np.asarray(value).tolist())


def _scalar_metrics(metrics):
    result = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"Metric {name!r} must be scalar; use log_array for arrays")
        result[name] = _json_value(array.item())
    return result


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, model_state):
    """Restore parameters, optimizer state, and step into an initialized TrainState."""
    return serialization.from_bytes(model_state, Path(path).read_bytes())


class Logger(ABC):
    def __init__(self, log_dir, checkpoint_filename="checkpoint.msgpack"):
        log_root = Path(log_dir).expanduser().resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        # Atomic directory creation also separates simultaneous cluster jobs.
        self.log_dir = Path(mkdtemp(prefix=f"run_{timestamp}_", dir=log_root))
        self.checkpoint_path = self.log_dir / checkpoint_filename

    @abstractmethod
    def log_parameters(self, parameters):
        pass

    @abstractmethod
    def log_metrics(self, metrics, step):
        pass

    @abstractmethod
    def log_figure(self, figure, figure_name, step):
        pass

    @abstractmethod
    def log_array(self, name, values, step):
        pass

    @abstractmethod
    def close(self, error=None):
        pass

    def save_checkpoint(self, model_state):
        _atomic_write(self.checkpoint_path, serialization.to_bytes(model_state))
        return self.checkpoint_path

    def log_evaluation(self, history, step):
        metrics = {}
        for name, values in history.items():
            if not values:
                continue
            value = values[-1]
            if name.startswith("figures/"):
                self.log_figure(value, name.removeprefix("figures/"), step)
            elif name.startswith("data/"):
                self.log_array(name.removeprefix("data/"), value, step)
            else:
                metrics[name] = value
        self.log_metrics(metrics, step)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(error=None if exc is None else f"{exc_type.__name__}: {exc}")


class JSONLogger(Logger):
    """Flush JSON Lines after every event; store figures and arrays beside the log.

    logs.jsonl is append-only, with one complete JSON object per line. run.json
    contains configuration and run status. Non-finite scalar metrics become null.
    Every instance creates its own run subdirectory beneath log_dir.
    With log_locally=False, only the replaceable checkpoint is written locally.
    """

    backend = "json"

    def __init__(self, log_dir, run_name="", checkpoint_filename="checkpoint.msgpack", log_locally=True):
        super().__init__(log_dir, checkpoint_filename)
        self.log_locally = log_locally
        self._closed = False
        self._metadata = {
            "format_version": 1,
            "backend": self.backend,
            "name": run_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        self._stream = ((self.log_dir / "logs.jsonl").open("x", encoding="utf-8")
                        if self.log_locally else None)
        self._write_metadata()

    def _write_metadata(self):
        if not self.log_locally:
            return
        data = json.dumps(self._metadata, indent=2, allow_nan=False).encode("utf-8")
        _atomic_write(self.log_dir / "run.json", data)

    def _record(self, event_type, **values):
        if not self.log_locally:
            return
        event = {"type": event_type, "time": datetime.now(timezone.utc).isoformat(), **values}
        self._stream.write(json.dumps(event, allow_nan=False) + "\n")
        self._stream.flush()

    def _artifact_path(self, folder, name, step, suffix):
        filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        path = self.log_dir / folder / f"{filename}_{int(step):08d}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_parameters(self, parameters):
        self._metadata["parameters"] = _json_value(parameters)
        self._write_metadata()

    def log_metrics(self, metrics, step):
        self._record("metrics", step=int(step), metrics=_scalar_metrics(metrics))

    def log_figure(self, figure, figure_name, step):
        if not self.log_locally:
            return None
        path = self._artifact_path("figures", figure_name, step, "png")
        figure.savefig(path, dpi=150, bbox_inches="tight")
        self._record("figure", name=figure_name, step=int(step), path=str(path.relative_to(self.log_dir)))
        return path

    def log_array(self, name, values, step):
        if not self.log_locally:
            return None
        path = self._artifact_path("data", name, step, "npz")
        values = np.asarray(values)
        np.savez_compressed(path, data=values)
        self._record(
            "array", name=name, step=int(step), path=str(path.relative_to(self.log_dir)),
            shape=list(values.shape), dtype=str(values.dtype),
        )
        return path

    def save_checkpoint(self, model_state):
        path = super().save_checkpoint(model_state)
        self._record("checkpoint", step=int(model_state.step), path=str(path.relative_to(self.log_dir)))
        return path

    def close(self, error=None):
        if self._closed:
            return
        self._metadata.update(
            status="finished" if error is None else "failed", error=error,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if self._stream is not None:
            self._stream.close()
        self._closed = True
        self._write_metadata()


class CometLogger(JSONLogger):
    """Send events to Comet, optionally retaining local JSON logs/artifacts."""

    backend = "comet"

    def __init__(self, log_dir, run_name="", checkpoint_filename="checkpoint.msgpack", log_locally=True, **comet_options):
        # JSON-only runs never import or initialize the Comet SDK.
        from comet_ml import Experiment

        super().__init__(log_dir, run_name, checkpoint_filename, log_locally=log_locally)
        try:
            self.experiment = Experiment(**{key: value for key, value in comet_options.items() if value is not None})
            self.experiment.set_name(run_name)
        except Exception as error:
            super().close(error=str(error))
            raise

    def log_parameters(self, parameters):
        super().log_parameters(parameters)
        self.experiment.log_parameters(flatten_dict(_json_value(parameters)))

    def log_metrics(self, metrics, step):
        super().log_metrics(metrics, step)
        # Comet converts bools to strings instead of numeric metrics.
        values = {
            name: int(value) if isinstance(value, bool) else value
            for name, value in _scalar_metrics(metrics).items() if value is not None
        }
        self.experiment.log_metrics(values, step=int(step))

    def log_figure(self, figure, figure_name, step):
        path = super().log_figure(figure, figure_name, step)
        self.experiment.log_figure(figure=figure, figure_name=figure_name, step=int(step))
        return path

    def log_array(self, name, values, step):
        path = super().log_array(name, values, step)
        if path is not None:
            self.experiment.log_asset(str(path), file_name=path.name, step=int(step))
        else:
            # Send the same NPZ payload without retaining an array file on disk.
            with io.BytesIO() as stream:
                np.savez_compressed(stream, data=np.asarray(values))
                filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
                self.experiment.log_asset_data(
                    stream.getvalue(), name=f"{filename}_{int(step):08d}.npz", step=int(step),
                )
        return path

    def close(self, error=None):
        if self._closed:
            return
        try:
            self.experiment.log_other("error", error)
            self.experiment.end()
        finally:
            super().close(error)


def create_logger(backend, log_dir, run_name="", checkpoint_filename="checkpoint.msgpack", comet_options=None, log_locally=True):
    if backend == "json":
        return JSONLogger(log_dir, run_name, checkpoint_filename, log_locally=log_locally)
    if backend == "comet":
        return CometLogger(log_dir, run_name, checkpoint_filename,
                           log_locally=log_locally, **(comet_options or {}))
    raise ValueError(f"Unknown logger: {backend!r}; choose 'json' or 'comet'")
