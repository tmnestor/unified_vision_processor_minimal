"""Memory Monitoring Utilities for Model Comparison
================================================

Provides comprehensive memory usage tracking for GPU and system memory
during model comparison operations.
"""

import gc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
from rich.console import Console

from .logging_config import VisionProcessorLogger


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific time."""

    timestamp: float
    label: str

    # System memory (RAM)
    system_total_gb: float
    system_used_gb: float
    system_available_gb: float
    system_percent: float

    # Process memory
    process_memory_gb: float
    process_memory_percent: float

    # GPU memory (if available)
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None


class MemoryMonitor:
    """Monitor and track memory usage throughout model operations."""

    def __init__(self, console: Optional[Console] = None, config=None):
        """Initialize memory monitor.

        Args:
            console: Rich console for output (optional)
            config: Configuration manager for logging (optional)
        """
        self.console = console or Console()
        self.config = config
        self.logger = VisionProcessorLogger(config)
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = time.time()

        # Check GPU availability
        self.gpu_available = self._check_gpu_available()

    def _check_gpu_available(self) -> bool:
        """Check if GPU memory monitoring is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_gpu_memory(
        self,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Get GPU memory usage in GB.

        Returns:
            Tuple of (used_gb, total_gb, reserved_gb) or (None, None, None) if not available
        """
        if not self.gpu_available:
            return None, None, None

        try:
            import torch

            # Get memory stats for GPU 0 (primary GPU)
            used_bytes = torch.cuda.memory_allocated(0)
            reserved_bytes = torch.cuda.memory_reserved(0)

            # Get total memory for GPU 0 specifically
            total_bytes = torch.cuda.get_device_properties(0).total_memory

            # Debug: Check if memory is spread across multiple GPUs
            if torch.cuda.device_count() > 1:
                total_used_all_gpus = sum(
                    torch.cuda.memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                )
                if total_used_all_gpus != used_bytes:
                    # Memory is spread across multiple GPUs - this is a problem for V100!
                    self.logger.warning(
                        f"Memory detected on multiple GPUs! Total across all GPUs: {total_used_all_gpus / (1024**3):.1f}GB"
                    )

            # Convert to GB
            used_gb = used_bytes / (1024**3)
            reserved_gb = reserved_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)

            return used_gb, total_gb, reserved_gb

        except Exception:
            return None, None, None

    def _get_system_memory(self) -> tuple[float, float, float, float]:
        """Get system memory usage.

        Returns:
            Tuple of (total_gb, used_gb, available_gb, percent_used)
        """
        memory = psutil.virtual_memory()

        total_gb = memory.total / (1024**3)
        used_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        percent = memory.percent

        return total_gb, used_gb, available_gb, percent

    def _get_process_memory(self) -> tuple[float, float]:
        """Get current process memory usage.

        Returns:
            Tuple of (memory_gb, memory_percent)
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        memory_gb = memory_info.rss / (1024**3)  # Resident Set Size in GB
        memory_percent = process.memory_percent()

        return memory_gb, memory_percent

    def take_snapshot(self, label: str) -> MemorySnapshot:
        """Take a memory usage snapshot.

        Args:
            label: Descriptive label for this snapshot

        Returns:
            MemorySnapshot with current memory state
        """
        # Get system memory
        sys_total, sys_used, sys_available, sys_percent = self._get_system_memory()

        # Get process memory
        proc_memory, proc_percent = self._get_process_memory()

        # Get GPU memory
        gpu_used, gpu_total, gpu_reserved = self._get_gpu_memory()
        gpu_percent = (gpu_used / gpu_total * 100) if gpu_used and gpu_total else None

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            label=label,
            system_total_gb=sys_total,
            system_used_gb=sys_used,
            system_available_gb=sys_available,
            system_percent=sys_percent,
            process_memory_gb=proc_memory,
            process_memory_percent=proc_percent,
            gpu_memory_used_gb=gpu_used,
            gpu_memory_total_gb=gpu_total,
            gpu_memory_percent=gpu_percent,
            gpu_memory_reserved_gb=gpu_reserved,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def print_snapshot(self, snapshot: MemorySnapshot, show_details: bool = False):
        """Print a memory snapshot to console.

        Args:
            snapshot: Memory snapshot to print
            show_details: Whether to show detailed breakdown
        """
        elapsed = snapshot.timestamp - self.start_time

        if show_details:
            self.console.print(
                f"\n📊 [bold]Memory Snapshot: {snapshot.label}[/bold] (t={elapsed:.1f}s)"
            )
            self.console.print(
                f"   🖥️  System RAM: {snapshot.system_used_gb:.1f}GB / {snapshot.system_total_gb:.1f}GB ({snapshot.system_percent:.1f}%)"
            )
            self.console.print(
                f"   🔧 Process RAM: {snapshot.process_memory_gb:.1f}GB ({snapshot.process_memory_percent:.1f}%)"
            )

            if (
                snapshot.gpu_memory_used_gb is not None
                and snapshot.gpu_memory_total_gb is not None
            ):
                # Calculate percentage based on reserved memory (what PyTorch actually claims)
                reserved_gb = (
                    snapshot.gpu_memory_reserved_gb or snapshot.gpu_memory_used_gb
                )
                reserved_percent = (
                    (reserved_gb / snapshot.gpu_memory_total_gb * 100)
                    if snapshot.gpu_memory_total_gb
                    else 0.0
                )

                self.console.print(
                    f"   🎮 GPU Reserved: {reserved_gb:.1f}GB / {snapshot.gpu_memory_total_gb:.1f}GB ({reserved_percent:.1f}%)"
                )
                self.console.print(
                    f"   📊 GPU Allocated: {snapshot.gpu_memory_used_gb:.1f}GB (active tensors)"
                )

                # V100 Production Warning - fail fast if config not available
                if not self.config or not hasattr(self.config, "memory_config"):
                    # Skip V100 warning if no config - memory monitor is optional diagnostic tool
                    pass
                else:
                    v100_limit = self.config.memory_config.v100_limit_gb
                    if reserved_gb > v100_limit:
                        self.console.print(
                            f"   ⚠️  WARNING: Reserved memory ({reserved_gb:.1f}GB) exceeds V100 limit ({v100_limit}GB)",
                            style="bold red",
                        )
        else:
            # Compact format
            gpu_info = (
                f", GPU: {snapshot.gpu_memory_used_gb:.1f}GB"
                if snapshot.gpu_memory_used_gb is not None
                else ""
            )
            self.console.print(
                f"💾 {snapshot.label}: RAM {snapshot.process_memory_gb:.1f}GB{gpu_info}"
            )

    def print_current_usage(self, label: str = "Current"):
        """Print current memory usage.

        Args:
            label: Label for the current measurement
        """
        snapshot = self.take_snapshot(label)
        self.print_snapshot(snapshot, show_details=True)

    def print_peak_usage(self):
        """Print peak memory usage across all snapshots."""
        if not self.snapshots:
            self.console.print("⚠️  No memory snapshots available")
            return

        # Find peaks
        peak_system = max(self.snapshots, key=lambda s: s.system_used_gb)
        peak_process = max(self.snapshots, key=lambda s: s.process_memory_gb)

        if any(s.gpu_memory_used_gb for s in self.snapshots):
            peak_gpu = max(
                (s for s in self.snapshots if s.gpu_memory_used_gb is not None),
                key=lambda s: s.gpu_memory_used_gb or 0.0,
            )
        else:
            peak_gpu = None

        self.console.print("\n🏔️  [bold]Peak Memory Usage:[/bold]")
        self.console.print(
            f"   🖥️  System: {peak_system.system_used_gb:.1f}GB ({peak_system.system_percent:.1f}%) at '{peak_system.label}'"
        )
        self.console.print(
            f"   🔧 Process: {peak_process.process_memory_gb:.1f}GB ({peak_process.process_memory_percent:.1f}%) at '{peak_process.label}'"
        )

        if peak_gpu:
            gpu_percent = peak_gpu.gpu_memory_percent or 0.0
            self.console.print(
                f"   🎮 GPU: {peak_gpu.gpu_memory_used_gb:.1f}GB ({gpu_percent:.1f}%) at '{peak_gpu.label}'"
            )

    def print_memory_timeline(self):
        """Print a timeline of memory usage."""
        if not self.snapshots:
            self.console.print("⚠️  No memory snapshots available")
            return

        self.console.print("\n📈 [bold]Memory Timeline:[/bold]")

        for snapshot in self.snapshots:
            elapsed = snapshot.timestamp - self.start_time
            gpu_info = (
                f" GPU:{snapshot.gpu_memory_used_gb:.1f}GB"
                if snapshot.gpu_memory_used_gb is not None
                else ""
            )

            self.console.print(
                f"   {elapsed:6.1f}s: {snapshot.label:<25} "
                f"RAM:{snapshot.process_memory_gb:.1f}GB{gpu_info}"
            )

    def cleanup_and_measure(self, label: str) -> MemorySnapshot:
        """Run garbage collection and take memory snapshot.

        Args:
            label: Label for the measurement

        Returns:
            Memory snapshot after cleanup
        """
        # Force garbage collection
        gc.collect()

        # GPU memory cleanup if available
        if self.gpu_available:
            try:
                import torch

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        # Take snapshot after cleanup
        return self.take_snapshot(label)

    def get_memory_summary(self) -> Dict[str, float]:
        """Get summary of memory usage across all snapshots.

        Returns:
            Dictionary with memory usage summary statistics
        """
        if not self.snapshots:
            return {}

        # Calculate statistics
        process_memories = [s.process_memory_gb for s in self.snapshots]
        system_memories = [s.system_used_gb for s in self.snapshots]
        gpu_memories = [
            s.gpu_memory_used_gb for s in self.snapshots if s.gpu_memory_used_gb
        ]

        summary = {
            "peak_process_memory_gb": max(process_memories),
            "avg_process_memory_gb": sum(process_memories) / len(process_memories),
            "peak_system_memory_gb": max(system_memories),
            "total_snapshots": len(self.snapshots),
            "monitoring_duration_s": self.snapshots[-1].timestamp
            - self.snapshots[0].timestamp,
        }

        if gpu_memories:
            summary.update(
                {
                    "peak_gpu_memory_gb": max(gpu_memories),
                    "avg_gpu_memory_gb": sum(gpu_memories) / len(gpu_memories),
                    "gpu_total_memory_gb": self.snapshots[0].gpu_memory_total_gb or 0.0,
                }
            )

        return summary

    def reset_snapshots(self) -> None:
        """Reset memory snapshots for new model measurement.

        This should be called between model comparisons to ensure
        each model gets independent memory measurements.
        """
        self.snapshots.clear()
        self.start_time = time.time()
        gc.collect()  # Clean up before starting fresh measurements

    def validate_memory_measurements(
        self, model_name: str, model_size_gb: float = None
    ) -> dict[str, bool]:
        """Validate memory measurements for logical consistency.

        Args:
            model_name: Name of the model being validated
            model_size_gb: Expected model size in GB (optional)

        Returns:
            Dictionary with validation results and flags
        """
        if not self.snapshots:
            return {
                "sufficient_snapshots": False,
                "has_peak_measurements": False,
                "logical_memory_progression": False,
                "adequate_monitoring_duration": False,
                "validation_summary": "No snapshots available for validation",
            }

        summary = self.get_memory_summary()

        # Check if we have sufficient snapshots
        sufficient_snapshots = summary.get("total_snapshots", 0) >= 3

        # Check if monitoring duration is adequate
        adequate_duration = summary.get("monitoring_duration_s", 0) > 10.0

        # Check if we captured peak measurements (peak > first measurement)
        if len(self.snapshots) >= 2:
            first_memory = self.snapshots[0].process_memory_gb
            peak_memory = summary.get("peak_process_memory_gb", 0)
            has_peak_measurements = (
                peak_memory > first_memory * 1.1
            )  # At least 10% increase
        else:
            has_peak_measurements = False

        # Check logical memory progression (should increase during processing)
        logical_progression = True
        if len(self.snapshots) >= 3:
            memory_values = [s.process_memory_gb for s in self.snapshots]
            # Memory should generally increase or stay stable, not decrease significantly
            for i in range(1, len(memory_values)):
                if memory_values[i] < memory_values[0] * 0.8:  # More than 20% decrease
                    logical_progression = False
                    break

        validation_flags = {
            "sufficient_snapshots": sufficient_snapshots,
            "has_peak_measurements": has_peak_measurements,
            "logical_memory_progression": logical_progression,
            "adequate_monitoring_duration": adequate_duration,
        }

        # Generate validation summary
        issues = []
        if not sufficient_snapshots:
            issues.append(
                f"Insufficient snapshots ({summary.get('total_snapshots', 0)}/3+ required)"
            )
        if not adequate_duration:
            issues.append(
                f"Short monitoring duration ({summary.get('monitoring_duration_s', 0):.1f}s)"
            )
        if not has_peak_measurements:
            issues.append("No inference peak captured")
        if not logical_progression:
            issues.append("Illogical memory progression detected")

        if issues:
            validation_flags["validation_summary"] = f"⚠️ Issues: {'; '.join(issues)}"
        else:
            validation_flags["validation_summary"] = (
                f"✅ {model_name} memory measurements validated"
            )

        return validation_flags

    def compare_model_memory_logic(
        self, model_summaries: dict[str, dict]
    ) -> dict[str, any]:
        """Compare memory measurements across models for logical consistency.

        Args:
            model_summaries: Dictionary of model_name -> memory_summary

        Returns:
            Dictionary with cross-model validation results
        """
        if len(model_summaries) < 2:
            return {"cross_model_validation": "Need at least 2 models for comparison"}

        # Extract model names and their memory usage
        model_data = []
        for model_name, summary in model_summaries.items():
            model_data.append(
                {
                    "name": model_name,
                    "peak_memory": summary.get("peak_process_memory_gb", 0),
                    "peak_gpu": summary.get("peak_gpu_memory_gb", 0),
                    "snapshots": summary.get("total_snapshots", 0),
                }
            )

        # Sort by model name to get consistent ordering
        model_data.sort(key=lambda x: x["name"])

        validation_results = {}

        # Check if larger models use more memory (based on naming convention)
        if len(model_data) == 2:
            # Assume models with "11b" or larger numbers should use more memory
            llama_model = next(
                (m for m in model_data if "llama" in m["name"].lower()), None
            )
            internvl_model = next(
                (m for m in model_data if "internvl" in m["name"].lower()), None
            )

            if llama_model and internvl_model:
                # Llama (11B) should use more process memory than InternVL (2B)
                llama_peak = llama_model["peak_memory"]
                internvl_peak = internvl_model["peak_memory"]

                logical_size_correlation = llama_peak > internvl_peak
                validation_results["logical_size_correlation"] = (
                    logical_size_correlation
                )

                if logical_size_correlation:
                    validation_results["size_correlation_summary"] = (
                        f"✅ Logical: Llama-11B ({llama_peak:.2f}GB) > InternVL-2B ({internvl_peak:.2f}GB)"
                    )
                else:
                    validation_results["size_correlation_summary"] = (
                        f"❌ ILLOGICAL: Llama-11B ({llama_peak:.2f}GB) < InternVL-2B ({internvl_peak:.2f}GB)"
                    )

        # Check that all models have adequate monitoring
        inadequate_models = [m["name"] for m in model_data if m["snapshots"] < 3]
        if inadequate_models:
            validation_results["inadequate_monitoring"] = inadequate_models
        else:
            validation_results["adequate_monitoring"] = (
                "✅ All models have sufficient snapshots"
            )

        return validation_results
