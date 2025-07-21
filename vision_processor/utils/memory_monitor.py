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

    def __init__(self, console: Optional[Console] = None):
        """Initialize memory monitor.

        Args:
            console: Rich console for output (optional)
        """
        self.console = console or Console()
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

    def _get_gpu_memory(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
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

            # Get total memory
            total_bytes = torch.cuda.get_device_properties(0).total_memory

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
            self.console.print(f"\n📊 [bold]Memory Snapshot: {snapshot.label}[/bold] (t={elapsed:.1f}s)")
            self.console.print(f"   🖥️  System RAM: {snapshot.system_used_gb:.1f}GB / {snapshot.system_total_gb:.1f}GB ({snapshot.system_percent:.1f}%)")
            self.console.print(f"   🔧 Process RAM: {snapshot.process_memory_gb:.1f}GB ({snapshot.process_memory_percent:.1f}%)")

            if snapshot.gpu_memory_used_gb is not None:
                self.console.print(f"   🎮 GPU Memory: {snapshot.gpu_memory_used_gb:.1f}GB / {snapshot.gpu_memory_total_gb:.1f}GB ({snapshot.gpu_memory_percent:.1f}%)")
                if snapshot.gpu_memory_reserved_gb:
                    self.console.print(f"   📦 GPU Reserved: {snapshot.gpu_memory_reserved_gb:.1f}GB")
        else:
            # Compact format
            gpu_info = f", GPU: {snapshot.gpu_memory_used_gb:.1f}GB" if snapshot.gpu_memory_used_gb else ""
            self.console.print(f"💾 {snapshot.label}: RAM {snapshot.process_memory_gb:.1f}GB{gpu_info}")

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
                (s for s in self.snapshots if s.gpu_memory_used_gb),
                key=lambda s: s.gpu_memory_used_gb
            )
        else:
            peak_gpu = None

        self.console.print("\n🏔️  [bold]Peak Memory Usage:[/bold]")
        self.console.print(f"   🖥️  System: {peak_system.system_used_gb:.1f}GB ({peak_system.system_percent:.1f}%) at '{peak_system.label}'")
        self.console.print(f"   🔧 Process: {peak_process.process_memory_gb:.1f}GB ({peak_process.process_memory_percent:.1f}%) at '{peak_process.label}'")

        if peak_gpu:
            self.console.print(f"   🎮 GPU: {peak_gpu.gpu_memory_used_gb:.1f}GB ({peak_gpu.gpu_memory_percent:.1f}%) at '{peak_gpu.label}'")

    def print_memory_timeline(self):
        """Print a timeline of memory usage."""
        if not self.snapshots:
            self.console.print("⚠️  No memory snapshots available")
            return

        self.console.print("\n📈 [bold]Memory Timeline:[/bold]")

        for snapshot in self.snapshots:
            elapsed = snapshot.timestamp - self.start_time
            gpu_info = f" GPU:{snapshot.gpu_memory_used_gb:.1f}GB" if snapshot.gpu_memory_used_gb else ""

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
        gpu_memories = [s.gpu_memory_used_gb for s in self.snapshots if s.gpu_memory_used_gb]

        summary = {
            "peak_process_memory_gb": max(process_memories),
            "avg_process_memory_gb": sum(process_memories) / len(process_memories),
            "peak_system_memory_gb": max(system_memories),
            "total_snapshots": len(self.snapshots),
            "monitoring_duration_s": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
        }

        if gpu_memories:
            summary.update({
                "peak_gpu_memory_gb": max(gpu_memories),
                "avg_gpu_memory_gb": sum(gpu_memories) / len(gpu_memories),
                "gpu_total_memory_gb": self.snapshots[0].gpu_memory_total_gb,
            })

        return summary
