"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lightning resource telemetry and run notifications
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) Track fit emissions with CodeCarbon.
2) Estimate registered-operation FLOPs for each observed batch shape.
3) Measure unprofiled CUDA step time and training allocator peaks.
4) Optionally report failed or completed fits to Telegram.

FLOPs include convolution backward and cuDNN recurrent matrix products; unsupported
elementwise operations and optimizer kernels are excluded. Summaries use flops.json
or flops-resume-<timestamp_ns>.json for resumed fits. CodeCarbon appends emissions.csv.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

import torch
from lightning.pytorch.callbacks import Callback
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode, register_flop_formula


# ── Config ───────────────────────────────────────────
@dataclass
class TelemetryConfig:
    warmup_steps: int = 20
    profile_steps: int = 4
    timing_steps: int = 20
    carbon_measure_power_secs: int = 15


# ── Carbon ───────────────────────────────────────────
class CarbonEmissionCallback(Callback):
    """Track fit emissions and persist final energy/emissions in the logger and CodeCarbon's `emissions.csv`."""

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.tracker = None

    def on_fit_start(self, trainer, pl_module):
        if not self.cfg.telemetry_enabled or not trainer.is_global_zero:
            return
        from codecarbon import EmissionsTracker

        self.tracker = EmissionsTracker(output_dir=str(self.output_dir), project_name=self.cfg.experiment_name,
                                        measure_power_secs=self.cfg.telemetry.carbon_measure_power_secs, save_to_api=False, log_level="error")
        self.tracker.start()

    def on_fit_end(self, trainer, pl_module):
        self._finish(trainer)

    def on_exception(self, trainer, pl_module, exception):
        self._finish(trainer)

    def _finish(self, trainer):
        if self.tracker is None:
            return
        tracker, self.tracker = self.tracker, None
        tracker.stop()
        data = tracker.final_emissions_data
        trainer.logger.log_metrics({"carbon/emissions_kg": data.emissions, "carbon/energy_kwh": data.energy_consumed}, step=trainer.global_step)


# ── Profiling ────────────────────────────────────────
def rnn_gate_flops(input_shape, weight_shapes):
    """Count recurrent matrix products at two FLOPs per multiply-add; exclude gate elementwise operations."""
    return 2 * math.prod(input_shape[:-1]) * sum(math.prod(shape) for shape in weight_shapes if len(shape) == 2)


@register_flop_formula(torch.ops.aten._cudnn_rnn)
def cudnn_rnn_flop(input_shape, weight_shapes, *args, out_shape=None, **kwargs):
    return rnn_gate_flops(input_shape, weight_shapes)


@register_flop_formula(torch.ops.aten._cudnn_rnn_backward)
def cudnn_rnn_backward_flop(input_shape, weight_shapes, *args, out_shape=None, **kwargs):
    return 2 * rnn_gate_flops(input_shape, weight_shapes)  # input and weight gradients each repeat the forward matmuls


def batch_signature(batch):
    """Shape key over all batch tensors and the leading batch size; FLOP counts are reused per key."""
    tensors = [leaf for leaf in tree_leaves(batch) if isinstance(leaf, torch.Tensor)]
    return ",".join("x".join(map(str, tensor.shape)) for tensor in tensors), len(tensors[0])


class FlopsProfilerCallback(Callback):
    """Count model operations and measure independent, unprofiled CUDA step spans."""

    def __init__(self, cfg, output_dir, resume=False):
        self.cfg, self.settings = cfg, cfg.telemetry
        filename = f"flops-resume-{time.time_ns()}.json" if resume else "flops.json"
        self.output_path = Path(output_dir) / filename
        self.counter = None
        self.events = []
        self.flops_by_shape = {}
        self.steps_by_shape = {}
        self.measured_steps = 0
        self.training_steps = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.timing_seconds = None
        self.timing_active = False

    def on_train_start(self, trainer, pl_module):
        if not self.cfg.telemetry_enabled:
            return
        self.device = pl_module.device

    def on_train_epoch_start(self, trainer, pl_module):
        if self.cfg.telemetry_enabled:
            self.epoch_start = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.cfg.telemetry_enabled:
            return
        torch.cuda.reset_peak_memory_stats(self.device)
        step = self.training_steps
        self.current_shape, batch_size = batch_signature(batch)
        self.steps_by_shape[self.current_shape] = self.steps_by_shape.get(self.current_shape, 0) + 1
        self.timing_active = False
        start = self.settings.warmup_steps
        if step >= start and (self.measured_steps < self.settings.profile_steps or self.current_shape not in self.flops_by_shape):
            self.counter = FlopCounterMode(display=False)
            self.counter.__enter__()
        elif step >= start and len(self.events) < self.settings.timing_steps:
            # Pairs exclude intervening validation, previews, and data loading.
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            self.events.append((begin, end, self.current_shape, batch_size))
            self.timing_active = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.cfg.telemetry_enabled:
            return
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(self.device))
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(self.device))
        if self.counter is not None:
            self._close_counter()
        elif self.timing_active:
            self.events[-1][1].record()
            if len(self.events) == self.settings.timing_steps:
                self.events[-1][1].synchronize()
                self.timing_seconds = sum(begin.elapsed_time(end) for begin, end, _, _ in self.events) / 1000
        self.training_steps += 1

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.cfg.telemetry_enabled:
            return
        metrics = self._metrics()
        metrics["epoch_wall_seconds"] = time.perf_counter() - self.epoch_start
        dashboard = {f"resources/{key}": metrics[key] for key in ("peak_memory_allocated_gib", "epoch_wall_seconds")}
        if "model_training_tflops_per_second" in metrics:
            dashboard["resources/model_training_tflops_per_second"] = metrics["model_training_tflops_per_second"]
        trainer.logger.log_metrics(dashboard, trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        if not self.cfg.telemetry_enabled:
            return
        summary = self._metrics()
        summary["accounting"] = "Registered model FLOPs, forward + backward; excludes unsupported operations and optimizer kernels"
        summary["timing"] = "Unprofiled CUDA spans with matching batch-shape FLOP counts; excludes validation, previews, and data loading"
        summary["profiled_batch_shapes"] = sorted(self.flops_by_shape)
        summary["training_steps_by_shape"] = self.steps_by_shape
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(summary, indent=2))

    def on_exception(self, trainer, pl_module, exception):
        if self.counter is not None:
            self._close_counter()

    def _close_counter(self):
        counter, self.counter = self.counter, None
        counter.__exit__(None, None, None)
        self.flops_by_shape[self.current_shape] = counter.get_total_flops()
        self.measured_steps += 1

    def _metrics(self):
        metrics = {"training_steps": self.training_steps, "profiled_steps": self.measured_steps,
                   "peak_memory_allocated_gib": self.peak_allocated / 2**30,
                   "peak_memory_reserved_gib": self.peak_reserved / 2**30}
        unprofiled_steps = sum(count for shape, count in self.steps_by_shape.items() if shape not in self.flops_by_shape)
        metrics["steps_with_unprofiled_shape"] = unprofiled_steps
        if self.measured_steps and not unprofiled_steps:
            total = sum(count * self.flops_by_shape[shape] for shape, count in self.steps_by_shape.items())
            per_step = total / self.training_steps
            metrics.update(model_flops_per_step=per_step, estimated_model_flops_total=per_step * self.training_steps)
            if self.timing_seconds is not None:
                seconds = self.timing_seconds / len(self.events)
                timed_flops = sum(self.flops_by_shape[shape] for _, _, shape, _ in self.events)
                examples = sum(batch_size for _, _, _, batch_size in self.events)
                metrics.update(gpu_step_seconds=seconds, model_training_tflops_per_second=timed_flops / self.timing_seconds / 1e12,
                               examples_per_gpu_second=examples / self.timing_seconds)
        return metrics


# ── Notifications ────────────────────────────────────
class TelegramCallback(Callback):
    """Report failed or completed fits to Telegram, optionally with a summary image.

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Credentials are checked at
    fit start; subsequent notification failures emit warnings.
    """

    def __init__(self, run):
        self.run = run
        self.token, self.chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
        self.start = time.monotonic()  # include failures before the first fit hook

    def on_fit_start(self, trainer, pl_module):
        self._call("getChat", {"chat_id": self.chat_id})

    def on_exception(self, trainer, pl_module, exception):
        self.report(f"failed: {type(exception).__name__}: {str(exception)[:500]}")

    def report(self, text, image=None):
        """Send the run header and text, attaching the PNG at `image` when supplied."""
        message = f"{self.run} ({(time.monotonic() - self.start) / 60:.0f} min)\n{text}"
        fields = {"chat_id": self.chat_id, "caption" if image else "text": message}
        try:
            self._call("sendPhoto" if image else "sendMessage", fields, image)
        except (urllib.error.URLError, TimeoutError) as error:  # notification failures do not interrupt training
            print(f"[WARNING] Telegram notification not sent: {error}")

    def _call(self, method, fields, image=None):
        """POST `fields`, plus the PNG at path `image` as `photo`, to the Bot API as multipart form data."""
        boundary = uuid.uuid4().hex
        body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode() for key, value in fields.items())
        if image:
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="{image.name}"\r\nContent-Type: image/png\r\n\r\n'.encode()
            body += image.read_bytes() + b"\r\n"
        request = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/{method}", body + f"--{boundary}--\r\n".encode(),
                                         {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        urllib.request.urlopen(request, timeout=30).close()
