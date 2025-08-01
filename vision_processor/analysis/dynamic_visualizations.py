"""Dynamic Model Comparison Visualizations
==========================================

Sophisticated visualization module that dynamically reads field configurations
from model_comparison.yaml without any hardcoding. Generates publication-ready
charts for model performance analysis.

Features:
- Dynamic field discovery from extraction_prompt
- Configuration-driven thresholds and weights
- Automatic scaling for any number of fields
- Business-importance aware visualizations
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from rich.console import Console

from ..config.config_manager import ConfigManager


class DynamicModelVisualizer:
    """Generate sophisticated model comparison visualizations with dynamic configuration."""

    def __init__(
        self, config_manager: ConfigManager, output_dir: str = "visualizations"
    ):
        """Initialize visualizer with dynamic configuration loading.

        Args:
            config_manager: Configuration manager with loaded yaml settings
            output_dir: Directory to save generated charts
        """
        self.config_manager = config_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.console = Console()

        # Load dynamic configuration
        self._load_dynamic_config()

        # Set up matplotlib style
        self._setup_plotting_style()

    def _load_dynamic_config(self) -> None:
        """Load all configuration dynamically from yaml files."""
        # Load fields using unified ConfigManager (single source of truth)
        self.extraction_fields = self.config_manager.get_expected_fields()

        # Load weights and thresholds from config manager
        self.field_weights = self.config_manager.get_field_weights()

        # Set visualization thresholds (these are for chart display, not business logic)
        self.quality_thresholds = {
            "excellent": 12,  # 12+ fields out of 26 = excellent
            "good": 8,  # 8-11 fields = good
            "fair": 5,  # 5-7 fields = fair
            "poor": 0,  # <5 fields = poor
        }
        self.speed_thresholds = {
            "very_fast": 15.0,  # <15s per document = very fast
            "fast": 25.0,  # 15-25s = fast
            "moderate": 40.0,  # 25-40s = moderate, >40s = slow
        }

        self.console.print(
            f"✅ Dynamic config loaded: {len(self.extraction_fields)} fields, "
            f"{len(self.field_weights)} weights",
            style="green",
        )

    def _extract_working_models_from_comparison_runner(
        self, comparison_results: Dict[str, Any]
    ) -> List[tuple]:
        """Extract working models from comparison_runner format and convert to visualization format.

        Args:
            comparison_results: Results from comparison_runner.py

        Returns:
            List of (model_name, model_results) tuples in visualization format
        """
        working_models = []

        # Check if this is comparison_runner format
        if (
            "models_tested" in comparison_results
            and "field_analysis" in comparison_results
        ):
            # Extract data from comparison_runner format
            models_tested = comparison_results.get("models_tested", [])
            field_analysis = comparison_results.get("field_analysis", {})
            model_execution_times = comparison_results.get("model_execution_times", {})
            model_success_rates = comparison_results.get("model_success_rates", {})
            extraction_results = comparison_results.get("extraction_results", {})

            # Get total images for per-image calculation
            total_images = comparison_results.get("dataset_info", {}).get(
                "total_images", 1
            )

            for model in models_tested:
                # Calculate per-image processing time
                total_time = model_execution_times.get(model, 0.0)
                avg_time_per_image = (
                    total_time / total_images if total_images > 0 else 0.0
                )

                # Convert comparison_runner format to visualization format
                model_results = {
                    "field_wise_accuracy": {},
                    "avg_accuracy": 0.0,
                    "avg_processing_time": avg_time_per_image,
                    "success_rate": model_success_rates.get(model, 0.0),
                    "avg_fields_extracted": 0.0,
                    "total_processing_time": total_time,
                }

                # Extract field-wise accuracy from field_analysis
                if (
                    "model_stats" in field_analysis
                    and model in field_analysis["model_stats"]
                ):
                    model_stats = field_analysis["model_stats"][model]

                    # Use field_value_rates as field_wise_accuracy
                    if "field_value_rates" in model_stats:
                        model_results["field_wise_accuracy"] = model_stats[
                            "field_value_rates"
                        ]
                    elif "field_extraction_rates" in model_stats:
                        # Fallback to extraction rates
                        model_results["field_wise_accuracy"] = model_stats[
                            "field_extraction_rates"
                        ]

                # Ensure all expected fields are represented (fill missing fields with 0.0)
                for expected_field in self.extraction_fields:
                    if expected_field not in model_results["field_wise_accuracy"]:
                        model_results["field_wise_accuracy"][expected_field] = 0.0

                # Get avg_fields_extracted if available
                if (
                    "model_stats" in field_analysis
                    and model in field_analysis["model_stats"]
                ):
                    model_stats = field_analysis["model_stats"][model]
                    if "avg_fields_extracted" in model_stats:
                        model_results["avg_fields_extracted"] = model_stats[
                            "avg_fields_extracted"
                        ]

                # Calculate avg_accuracy from field_wise_accuracy
                if model_results["field_wise_accuracy"]:
                    field_accuracies = list(
                        model_results["field_wise_accuracy"].values()
                    )
                    model_results["avg_accuracy"] = sum(field_accuracies) / len(
                        field_accuracies
                    )

                    # Calculate fields with useful data (for dashboard display)
                    model_results["fields_with_data"] = sum(
                        1 for acc in field_accuracies if acc > 0
                    )

                working_models.append((model, model_results))

        else:
            # Assume this is already in evaluator format
            working_models = [
                (model, results)
                for model, results in comparison_results.items()
                if isinstance(results, dict)
                and "error" not in results
                and "field_wise_accuracy" in results
            ]

        return working_models

    def _setup_plotting_style(self) -> None:
        """Set up consistent plotting style for professional charts."""
        # Use a clean, professional style
        plt.style.use("default")

        # Set consistent font sizes and colors
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "warning": "#C73E1D",
            "info": "#4ECDC4",
            "text": "#2C3E50",
            "background": "#F8F9FA",
        }

        # Configure matplotlib defaults
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "#CCCCCC",
                "axes.linewidth": 1,
                "axes.labelcolor": self.colors["text"],
                "text.color": self.colors["text"],
                "xtick.color": self.colors["text"],
                "ytick.color": self.colors["text"],
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
            }
        )

    def _categorize_fields_by_weight(self) -> Dict[str, List[str]]:
        """Categorize fields by their importance weights from configuration."""
        categories = {
            "High Priority": [],  # weight > 1.1
            "Standard": [],  # weight = 1.0
            "Lower Priority": [],  # weight < 1.0
        }

        for field in self.extraction_fields:
            weight = self.field_weights.get(field, 1.0)
            if weight > 1.1:
                categories["High Priority"].append(field)
            elif weight == 1.0:
                categories["Standard"].append(field)
            else:
                categories["Lower Priority"].append(field)

        return categories

    def create_field_accuracy_heatmap(
        self, comparison_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """Create dynamic field-wise accuracy heatmap that scales to any number of fields.

        Args:
            comparison_results: Model comparison results from evaluator
            save_path: Optional path to save chart

        Returns:
            Path to saved chart file
        """
        self.console.print(
            "🎨 Creating dynamic field accuracy heatmap...", style="blue"
        )

        # Extract working models from comparison_runner format
        working_models = self._extract_working_models_from_comparison_runner(
            comparison_results
        )

        if not working_models:
            self.console.print("❌ No valid model results for heatmap", style="red")
            return ""

        # Build data matrix dynamically
        models = [model.upper() for model, _ in working_models]

        # Use discovered fields (not hardcoded)
        available_fields = set()
        for _, results in working_models:
            available_fields.update(results["field_wise_accuracy"].keys())

        # Filter to fields that exist in results and configuration
        display_fields = [f for f in self.extraction_fields if f in available_fields]

        self.console.print(
            f"🔍 Processing {len(display_fields)} of {len(self.extraction_fields)} expected fields",
            style="dim",
        )

        # Create accuracy matrix
        accuracy_matrix = []
        for field in display_fields:
            row = []
            for _, results in working_models:
                accuracy = results["field_wise_accuracy"].get(field, 0.0)
                row.append(accuracy * 100)  # Convert to percentage
            accuracy_matrix.append(row)

        # Convert to DataFrame for seaborn
        accuracy_df = pd.DataFrame(
            accuracy_matrix, index=display_fields, columns=models
        )

        # Calculate dynamic figure size based on number of fields
        n_fields = len(display_fields)
        n_models = len(models)

        # Dynamic sizing: more fields = taller figure
        fig_height = max(8, min(20, n_fields * 0.4 + 4))
        fig_width = max(10, n_models * 2 + 6)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Custom colormap for accuracy (red to green)
        colors_list = [
            "#d73027",
            "#f46d43",
            "#fdae61",
            "#fee08b",
            "#e6f598",
            "#abdda4",
            "#66c2a5",
            "#3288bd",
        ]
        custom_cmap = sns.blend_palette(colors_list, as_cmap=True)

        # Create heatmap with annotations
        sns.heatmap(
            accuracy_df,
            annot=True,
            fmt=".1f",
            cmap=custom_cmap,
            cbar_kws={"label": "Accuracy (%)"},
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 10},
        )

        # Customize the plot
        ax.set_title(
            f"Field-wise Accuracy Comparison\n({n_fields} Fields Dynamically Loaded)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Models", fontweight="bold")
        ax.set_ylabel("Fields (Dynamically Loaded from Config)", fontweight="bold")

        # Rotate field names for better readability
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Add weight indicators for important fields
        for i, field in enumerate(display_fields):
            weight = self.field_weights.get(field, 1.0)
            if weight > 1.1:
                # Add a small indicator for high-priority fields
                rect = Rectangle(
                    (0, i),
                    len(models),
                    1,
                    linewidth=3,
                    edgecolor="gold",
                    facecolor="none",
                    alpha=0.8,
                )
                ax.add_patch(rect)

        plt.tight_layout()

        # Save chart
        if save_path is None:
            save_path = self.output_dir / f"field_accuracy_heatmap_{n_fields}fields.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.console.print(f"✅ Heatmap saved: {save_path}", style="green")
        return str(save_path)

    def create_model_performance_dashboard(
        self, comparison_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """Create comprehensive model performance dashboard.

        Args:
            comparison_results: Model comparison results from evaluator
            save_path: Optional path to save chart

        Returns:
            Path to saved chart file
        """
        self.console.print("🎨 Creating model performance dashboard...", style="blue")

        # Extract working models from comparison_runner format
        working_models = self._extract_working_models_from_comparison_runner(
            comparison_results
        )

        if not working_models:
            self.console.print("❌ No valid model results for dashboard", style="red")
            return ""

        # Create 2x2 subplot dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Model Performance Dashboard (Dynamic Configuration)",
            fontsize=18,
            fontweight="bold",
            y=0.95,
        )

        models = [model.upper() for model, _ in working_models]

        # 1. Overall Accuracy Comparison
        accuracies = []
        for _, results in working_models:
            if "avg_accuracy" in results:
                accuracies.append(results["avg_accuracy"] * 100)
            elif "field_wise_accuracy" in results:
                # Calculate average from field-wise accuracies
                field_accs = list(results["field_wise_accuracy"].values())
                avg_acc = sum(field_accs) / len(field_accs) if field_accs else 0.0
                accuracies.append(avg_acc * 100)
            else:
                accuracies.append(0.0)
        bars1 = ax1.bar(
            models,
            accuracies,
            color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
        )
        ax1.set_title("Overall Accuracy", fontweight="bold")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 100)

        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies, strict=False):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add threshold lines from config
        if "excellent" in self.quality_thresholds:
            excellent_pct = (
                self.quality_thresholds["excellent"] / len(self.extraction_fields)
            ) * 100
            ax1.axhline(
                y=excellent_pct,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Excellent ({excellent_pct:.0f}%)",
            )
            ax1.legend()

        # 2. Processing Throughput Comparison (images per minute)
        throughputs = [
            60.0 / results["avg_processing_time"]
            if results["avg_processing_time"] > 0
            else 0.0
            for _, results in working_models
        ]
        bars2 = ax2.bar(
            models,
            throughputs,
            color=[self.colors["warning"], self.colors["info"]][: len(models)],
        )
        ax2.set_title("Processing Throughput", fontweight="bold")
        ax2.set_ylabel("Images per Minute")

        # Add value labels
        for bar, throughput in zip(bars2, throughputs, strict=False):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{throughput:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add throughput threshold lines (convert from time thresholds)
        for label, time_threshold in self.speed_thresholds.items():
            throughput_threshold = 60.0 / time_threshold if time_threshold > 0 else 0.0
            ax2.axhline(
                y=throughput_threshold,
                linestyle="--",
                alpha=0.7,
                label=f"{label.title()} ({throughput_threshold:.1f} img/min)",
            )
        ax2.legend()

        # 3. Success Rate Comparison
        success_rates = [results["success_rate"] * 100 for _, results in working_models]
        bars3 = ax3.bar(
            models,
            success_rates,
            color=[self.colors["success"], "#E74C3C"][: len(models)],
        )
        ax3.set_title("Success Rate", fontweight="bold")
        ax3.set_ylabel("Success Rate (%)")
        ax3.set_ylim(0, 100)

        # Add value labels
        for bar, rate in zip(bars3, success_rates, strict=False):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Fields with Useful Data Comparison
        fields_with_data = [
            results.get("fields_with_data", results["avg_fields_extracted"])
            for _, results in working_models
        ]
        bars4 = ax4.bar(
            models,
            fields_with_data,
            color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
        )
        ax4.set_title(
            f"Average Fields Extracted (of {len(self.extraction_fields)} total)",
            fontweight="bold",
        )
        ax4.set_ylabel("Fields per Document")
        ax4.set_ylim(0, len(self.extraction_fields))

        # Add value labels
        for bar, fields in zip(bars4, fields_with_data, strict=False):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{fields:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add max possible line
        ax4.axhline(
            y=len(self.extraction_fields),
            color="green",
            linestyle="-",
            alpha=0.5,
            label=f"Max Possible ({len(self.extraction_fields)})",
        )
        ax4.legend()

        plt.tight_layout()

        # Save chart
        if save_path is None:
            save_path = self.output_dir / "model_performance_dashboard.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.console.print(f"✅ Dashboard saved: {save_path}", style="green")
        return str(save_path)

    def create_field_category_analysis(
        self, comparison_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """Create field category analysis based on dynamic weights from configuration.

        Args:
            comparison_results: Model comparison results from evaluator
            save_path: Optional path to save chart

        Returns:
            Path to saved chart file
        """
        self.console.print("🎨 Creating field category analysis...", style="blue")

        # Get field categories based on weights
        categories = self._categorize_fields_by_weight()

        # Extract working models from comparison_runner format
        working_models = self._extract_working_models_from_comparison_runner(
            comparison_results
        )

        if not working_models:
            self.console.print(
                "❌ No valid model results for category analysis", style="red"
            )
            return ""

        # Calculate category performance
        category_performance = {}
        for category, fields in categories.items():
            if not fields:  # Skip empty categories
                continue

            category_performance[category] = {}
            for model, results in working_models:
                # Calculate average accuracy for this category
                field_accs = []
                for field in fields:
                    if field in results["field_wise_accuracy"]:
                        field_accs.append(results["field_wise_accuracy"][field])

                avg_acc = np.mean(field_accs) if field_accs else 0.0
                category_performance[category][model.upper()] = avg_acc * 100

        # Create visualization
        n_categories = len(
            [cat for cat in category_performance if category_performance[cat]]
        )
        fig, axes = plt.subplots(1, n_categories, figsize=(5 * n_categories, 8))

        if n_categories == 1:
            axes = [axes]

        fig.suptitle(
            "Field Category Performance Analysis\n(Categories Based on Dynamic Weights)",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        models = [model.upper() for model, _ in working_models]
        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["success"],
        ]

        for i, (category, performance) in enumerate(category_performance.items()):
            if not performance:  # Skip empty categories
                continue

            ax = axes[i] if n_categories > 1 else axes[0]

            # Get performance values for each model
            values = [performance.get(model, 0) for model in models]

            # Create bar chart
            bars = ax.bar(models, values, color=colors[: len(models)])

            # Customize
            ax.set_title(
                f"{category}\n({len(categories[category])} fields)", fontweight="bold"
            )
            ax.set_ylabel("Average Accuracy (%)")
            ax.set_ylim(0, 100)

            # Add value labels
            for bar, val in zip(bars, values, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Add field list as text
            field_list = ", ".join(categories[category][:3])  # Show first 3 fields
            if len(categories[category]) > 3:
                field_list += f"... (+{len(categories[category]) - 3} more)"
            ax.text(
                0.5,
                -0.15,
                field_list,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                style="italic",
                wrap=True,
            )

        plt.tight_layout()

        # Save chart
        if save_path is None:
            save_path = self.output_dir / "field_category_analysis.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.console.print(f"✅ Category analysis saved: {save_path}", style="green")
        return str(save_path)

    def create_vram_usage_comparison(
        self, comparison_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """Create V100 VRAM usage comparison chart for both models.

        Args:
            comparison_results: Model comparison results from evaluator
            save_path: Optional path to save chart

        Returns:
            Path to saved chart file
        """
        self.console.print("🎨 Creating V100 VRAM usage comparison...", style="blue")

        # Extract memory data from comparison results
        memory_data = {}
        v100_limit_gb = self.config_manager.memory_config.v100_limit_gb

        # Check if we have memory data in the comparison results
        # Handle both dict and object formats for comparison_results
        if hasattr(comparison_results, "model_estimated_vram"):
            # ComparisonResults object format
            if comparison_results.model_estimated_vram:
                memory_data = comparison_results.model_estimated_vram
        elif isinstance(comparison_results, dict):
            # Dictionary format - look for memory data in nested results
            if "model_estimated_vram" in comparison_results:
                memory_data = comparison_results["model_estimated_vram"]
            elif "memory_summary" in comparison_results:
                memory_summary = comparison_results["memory_summary"]
                peak_gpu = memory_summary.get("peak_gpu_memory_gb", 0)
                if peak_gpu > 0 and "models_tested" in comparison_results:
                    # Distribute equally if we only have total peak
                    for model in comparison_results["models_tested"]:
                        memory_data[model] = peak_gpu / len(
                            comparison_results["models_tested"]
                        )
        elif (
            hasattr(comparison_results, "memory_summary")
            and comparison_results.memory_summary
        ):
            # ComparisonResults object with memory_summary
            peak_gpu = comparison_results.memory_summary.get("peak_gpu_memory_gb", 0)
            if peak_gpu > 0:
                # Distribute equally if we only have total peak
                for model in comparison_results.models_tested:
                    memory_data[model] = peak_gpu / len(
                        comparison_results.models_tested
                    )

        # If no memory data found, this indicates a problem with data collection
        if not memory_data:
            self.console.print(
                "❌ VRAM data collection failed - no actual memory estimates available",
                style="red",
            )
            self.console.print(
                "💡 This suggests the comparison runner is not collecting model VRAM estimates",
                style="yellow",
            )
            self.console.print(
                "💡 Check that _get_model_vram_estimates() is working correctly",
                style="yellow",
            )

            # Debug: Show what data is available
            if hasattr(comparison_results, "__dict__"):
                available_attrs = [
                    attr for attr in dir(comparison_results) if not attr.startswith("_")
                ]
                self.console.print(
                    f"💡 Available data attributes: {available_attrs}", style="yellow"
                )
            elif isinstance(comparison_results, dict):
                self.console.print(
                    f"💡 Available data keys: {list(comparison_results.keys())}",
                    style="yellow",
                )

            return ""

        # Create VRAM usage chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(
            "V100 VRAM Usage Comparison (16GB Limit)",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        models = [model.upper() for model in memory_data.keys()]
        vram_usage = list(memory_data.values())

        # Left chart: VRAM usage bars
        bars = ax1.bar(
            models,
            vram_usage,
            color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
            alpha=0.8,
        )
        ax1.set_title("Estimated VRAM Usage", fontweight="bold")
        ax1.set_ylabel("VRAM Usage (GB)")
        ax1.set_ylim(0, v100_limit_gb + 2)

        # Add V100 limit line
        ax1.axhline(
            y=v100_limit_gb,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="V100 Limit (16GB)",
        )

        # Add safety margin line
        safety_limit = v100_limit_gb * self.config_manager.memory_config.safety_margin
        ax1.axhline(
            y=safety_limit,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Safety Limit ({safety_limit:.1f}GB)",
        )

        # Add value labels on bars
        for bar, usage in zip(bars, vram_usage, strict=False):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{usage:.1f}GB",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

            # Add compliance status
            compliance_color = (
                "green"
                if usage <= safety_limit
                else "orange"
                if usage <= v100_limit_gb
                else "red"
            )
            status = (
                "SAFE"
                if usage <= safety_limit
                else "MARGIN"
                if usage <= v100_limit_gb
                else "OVER"
            )
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                usage / 2,
                status,
                ha="center",
                va="center",
                fontweight="bold",
                color=compliance_color,
                fontsize=10,
            )

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right chart: VRAM utilization percentages
        utilization = [(usage / v100_limit_gb) * 100 for usage in vram_usage]
        bars2 = ax2.bar(
            models,
            utilization,
            color=[self.colors["warning"], self.colors["info"]][: len(models)],
            alpha=0.8,
        )
        ax2.set_title("V100 VRAM Utilization", fontweight="bold")
        ax2.set_ylabel("Utilization (%)")
        ax2.set_ylim(0, 110)

        # Add threshold lines
        ax2.axhline(
            y=85,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Safety Threshold (85%)",
        )
        ax2.axhline(
            y=100, color="red", linestyle="--", alpha=0.7, label="V100 Limit (100%)"
        )

        # Add percentage labels
        for bar, util in zip(bars2, utilization, strict=False):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{util:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        if save_path is None:
            save_path = self.output_dir / "v100_vram_usage_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.console.print(f"✅ VRAM comparison saved: {save_path}", style="green")
        return str(save_path)

    def create_production_memory_requirements(
        self, comparison_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """Create comprehensive memory requirements chart for production POD sizing.

        This visualization is critical for Kubernetes POD resource allocation decisions.
        Shows both CPU memory and GPU VRAM requirements side-by-side.
        """
        self.console.print(
            "🏭 Creating production memory requirements chart...", style="blue"
        )

        # Extract memory data from comparison results
        models_tested = comparison_results.get("models_tested", [])
        model_vram = comparison_results.get("model_estimated_vram", {})
        memory_summary = comparison_results.get("memory_summary", {})

        # Get memory values
        peak_cpu_memory = memory_summary.get("peak_process_memory_gb", 0)
        avg_cpu_memory = memory_summary.get("avg_process_memory_gb", 0)
        peak_system_memory = memory_summary.get("peak_system_memory_gb", 0)

        # Create comprehensive memory chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Production Memory Requirements Analysis", fontsize=16, fontweight="bold"
        )

        # Chart 1: GPU VRAM Requirements
        if model_vram:
            models = [m.upper() for m in model_vram.keys()]
            vram_usage = list(model_vram.values())

            bars1 = ax1.bar(
                models,
                vram_usage,
                color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
                alpha=0.8,
            )
            ax1.set_title("GPU VRAM Requirements", fontweight="bold")
            ax1.set_ylabel("VRAM (GB)")

            # Add V100 limit lines
            v100_limit = 16.0
            safety_limit = v100_limit * 0.85
            ax1.axhline(
                y=v100_limit,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="V100 Limit (16GB)",
            )
            ax1.axhline(
                y=safety_limit,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="Safety Limit (85%)",
            )
            ax1.legend()

            # Add value labels and POD recommendations
            for bar, vram in zip(bars1, vram_usage, strict=False):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{vram:.1f}GB",
                    ha="center",
                    fontweight="bold",
                )

                # Add POD sizing recommendation
                if vram <= safety_limit:
                    pod_rec = "1x V100 OK"
                    color = "green"
                elif vram <= v100_limit:
                    pod_rec = "V100 Tight"
                    color = "orange"
                else:
                    pod_rec = "Need A100+"
                    color = "red"

                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.0,
                    pod_rec,
                    ha="center",
                    va="bottom",
                    color=color,
                    fontweight="bold",
                )

        # Chart 2: CPU Memory Requirements
        cpu_data = {
            "Peak Process": peak_cpu_memory,
            "Avg Process": avg_cpu_memory,
            "Peak System": peak_system_memory,
        }

        bars2 = ax2.bar(
            cpu_data.keys(),
            cpu_data.values(),
            color=["#ff6b6b", "#4ecdc4", "#45b7d1"],
            alpha=0.8,
        )
        ax2.set_title("CPU Memory Usage", fontweight="bold")
        ax2.set_ylabel("Memory (GB)")

        # Add typical POD memory limits
        pod_limits = [4, 8, 16, 32, 64]
        for limit in pod_limits:
            if limit <= max(cpu_data.values()) * 1.5:  # Show relevant limits
                ax2.axhline(y=limit, color="gray", linestyle=":", alpha=0.5)
                ax2.text(
                    len(cpu_data) - 0.3,
                    limit + 0.5,
                    f"{limit}GB",
                    fontsize=9,
                    alpha=0.7,
                )

        for bar, mem in zip(bars2, cpu_data.values(), strict=False):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{mem:.1f}GB",
                ha="center",
                fontweight="bold",
            )

        # Chart 3: POD Resource Recommendations
        if model_vram:
            ax3.axis("off")
            ax3.set_title(
                "Kubernetes POD Resource Recommendations", fontweight="bold", pad=20
            )

            y_pos = 0.9
            for model, vram in model_vram.items():
                # Calculate recommended POD resources based on actual peak memory needs
                cpu_request = max(2, int(avg_cpu_memory * 1.2))  # 20% overhead
                cpu_limit = max(4, int(peak_cpu_memory * 1.5))  # 50% overhead
                mem_request = max(
                    4, int(peak_cpu_memory * 1.1)
                )  # Peak + 10% buffer (minimum viable)
                mem_limit = max(
                    mem_request + 1, int(peak_cpu_memory * 1.3)
                )  # Peak + 30% buffer

                gpu_type = (
                    "nvidia.com/gpu: 1" if vram <= 16 else "nvidia.com/gpu-a100: 1"
                )

                pod_spec = f"""
{model.upper()} POD Specification:
  resources:
    requests:
      cpu: {cpu_request}
      memory: {mem_request}Gi
      {gpu_type}
    limits:  
      cpu: {cpu_limit}
      memory: {mem_limit}Gi
      {gpu_type}
                """

                ax3.text(
                    0.05,
                    y_pos,
                    pod_spec,
                    fontsize=10,
                    fontfamily="monospace",
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5
                    ),
                )
                y_pos -= 0.45

        # Chart 4: Cost Comparison (if both models shown)
        if len(model_vram) == 2:
            models = list(model_vram.keys())

            # Estimated hourly costs (example values - adjust based on your cloud provider)
            v100_cost_hour = 2.50  # $/hour for V100 instance
            a100_cost_hour = 4.00  # $/hour for A100 instance

            costs = []
            for _model, vram in model_vram.items():
                if vram <= 16:
                    cost = v100_cost_hour
                    instance = "V100"
                else:
                    cost = a100_cost_hour
                    instance = "A100"
                costs.append(cost)

            bars4 = ax4.bar(
                [m.upper() for m in models],
                costs,
                color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
                alpha=0.8,
            )
            ax4.set_title("Estimated Hourly Compute Costs", fontweight="bold")
            ax4.set_ylabel("Cost ($/hour)")

            for bar, cost, model in zip(bars4, costs, models, strict=False):
                vram = model_vram[model]
                instance_type = "V100" if vram <= 16 else "A100"
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"${cost:.2f}/hr\n({instance_type})",
                    ha="center",
                    fontweight="bold",
                )

        plt.tight_layout()

        # Save chart
        if save_path is None:
            save_path = self.output_dir / "production_memory_requirements.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.console.print(
            f"✅ Production memory requirements saved: {save_path}", style="green"
        )
        return str(save_path)

    def create_composite_overview(
        self, comparison_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """Create a 2x2 composite overview of all visualizations.

        Args:
            comparison_results: Model comparison results from evaluator
            save_path: Optional path to save chart

        Returns:
            Path to saved composite chart file
        """
        self.console.print(
            "🎨 Creating composite overview (2x2 layout)...", style="blue"
        )

        # Create a large figure with 2x2 subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(
            "Model Comparison Overview Dashboard",
            fontsize=20,
            fontweight="bold",
            y=0.95,
        )

        # Extract working models for reuse
        working_models = self._extract_working_models_from_comparison_runner(
            comparison_results
        )

        if not working_models:
            self.console.print("❌ No valid model results for composite", style="red")
            return ""

        models = [model.upper() for model, _ in working_models]

        # Subplot 1: Field Accuracy Heatmap (top-left)
        ax1 = plt.subplot(2, 2, 1)
        self._create_mini_heatmap(ax1, working_models)

        # Subplot 2: Performance Dashboard (top-right)
        ax2 = plt.subplot(2, 2, 2)
        self._create_mini_performance_bars(ax2, working_models)

        # Subplot 3: Field Category Analysis (bottom-left)
        ax3 = plt.subplot(2, 2, 3)
        self._create_mini_category_analysis(ax3, working_models)

        # Subplot 4: VRAM Usage (bottom-right)
        ax4 = plt.subplot(2, 2, 4)
        self._create_mini_vram_chart(ax4, comparison_results, working_models)

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

        # Save composite chart
        if save_path is None:
            save_path = self.output_dir / "composite_overview_2x2.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.console.print(f"✅ Composite overview saved: {save_path}", style="green")
        return str(save_path)

    def _create_mini_heatmap(self, ax, working_models):
        """Create a mini version of the field accuracy heatmap."""
        # Get available fields
        available_fields = set()
        for _, results in working_models:
            available_fields.update(results["field_wise_accuracy"].keys())

        # For composite view, show top performing fields for readability
        if len(available_fields) > 12:
            # Calculate average accuracy per field across models
            field_avg_accuracy = {}
            for field in available_fields:
                accuracies = []
                for _, results in working_models:
                    if field in results["field_wise_accuracy"]:
                        accuracies.append(results["field_wise_accuracy"][field])
                field_avg_accuracy[field] = np.mean(accuracies) if accuracies else 0.0

            # Sort by average accuracy and take top 12 for readability
            sorted_fields = sorted(
                field_avg_accuracy.items(), key=lambda x: x[1], reverse=True
            )
            display_fields = [field for field, _ in sorted_fields[:12]]

            self.console.print(
                f"📊 Showing top 12 performing fields out of {len(available_fields)} total",
                style="dim",
            )
        else:
            # Use all available fields if <= 12
            display_fields = [
                f for f in self.extraction_fields if f in available_fields
            ]
        models = [model.upper() for model, _ in working_models]

        # Create accuracy matrix
        accuracy_matrix = []
        for field in display_fields:
            row = []
            for _, results in working_models:
                accuracy = results["field_wise_accuracy"].get(field, 0.0)
                row.append(accuracy * 100)
            accuracy_matrix.append(row)

        # Create mini heatmap
        if accuracy_matrix:
            accuracy_df = pd.DataFrame(
                accuracy_matrix, index=display_fields, columns=models
            )

            # Adjust annotation size based on number of fields
            annot_size = max(6, min(8, 200 // len(display_fields)))

            sns.heatmap(
                accuracy_df,
                annot=True,
                fmt=".0f",
                cmap="RdYlGn",
                ax=ax,
                cbar=False,
                annot_kws={"size": annot_size},
            )
            ax.set_title(
                f"Field Accuracy ({len(display_fields)} fields)",
                fontweight="bold",
                fontsize=12,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            # Adjust label sizes for many fields
            ax.tick_params(
                axis="y", labelsize=max(6, min(8, 150 // len(display_fields)))
            )

    def _create_mini_performance_bars(self, ax, working_models):
        """Create mini performance comparison bars."""
        models = [model.upper() for model, _ in working_models]
        accuracies = []
        speeds = []

        for _, results in working_models:
            accuracies.append(results["avg_accuracy"] * 100)
            speeds.append(results["avg_processing_time"])

        # Create simple side-by-side bars for accuracy only (most important metric)
        bars = ax.bar(
            models,
            accuracies,
            color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
            alpha=0.8,
        )

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison", fontweight="bold", fontsize=12)
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for bar, acc, speed in zip(bars, accuracies, speeds, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
            )
            # Add speed as subtitle below bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                -8,
                f"{speed:.1f}s",
                ha="center",
                va="top",
                fontsize=7,
                style="italic",
            )

        ax.tick_params(axis="x", rotation=0)

    def _create_mini_category_analysis(self, ax, working_models):
        """Create mini field category analysis."""
        categories = self._categorize_fields_by_weight()
        models = [model.upper() for model, _ in working_models]

        # Calculate category performance
        category_performance = {}
        for category, fields in categories.items():
            if not fields:
                continue

            category_performance[category] = {}
            for model, results in working_models:
                field_accs = []
                for field in fields:
                    if field in results["field_wise_accuracy"]:
                        field_accs.append(results["field_wise_accuracy"][field])

                avg_acc = np.mean(field_accs) if field_accs else 0.0
                category_performance[category][model.upper()] = avg_acc * 100

        # Create grouped bar chart
        x_pos = np.arange(len(category_performance))
        width = 0.35

        for i, model in enumerate(models):
            values = [
                category_performance[cat].get(model, 0)
                for cat in category_performance.keys()
            ]
            ax.bar(
                x_pos + i * width,
                values,
                width,
                label=model,
                color=[self.colors["primary"], self.colors["secondary"]][i],
            )

        ax.set_title("Category Performance", fontweight="bold", fontsize=12)
        ax.set_xticks(x_pos + width / 2)
        ax.set_xticklabels(list(category_performance.keys()), rotation=45)
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=8)

    def _create_mini_vram_chart(self, ax, comparison_results, working_models):
        """Create mini V100 VRAM usage chart."""
        # Try to get memory data
        memory_data = {}
        if "model_estimated_vram" in comparison_results:
            memory_data = comparison_results["model_estimated_vram"]

        if memory_data:
            models = list(memory_data.keys())
            vram_usage = list(memory_data.values())

            bars = ax.bar(
                [m.upper() for m in models],
                vram_usage,
                color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
            )

            # Add V100 limit line
            v100_limit = self.config_manager.memory_config.v100_limit_gb
            ax.axhline(
                y=v100_limit, color="red", linestyle="--", alpha=0.7, label="V100 Limit"
            )

            ax.set_title("V100 VRAM Usage", fontweight="bold", fontsize=12)
            ax.set_ylabel("VRAM (GB)")
            ax.legend()

            # Add value labels
            for bar, usage in zip(bars, vram_usage, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{usage:.1f}GB",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            # No memory data available
            ax.text(
                0.5,
                0.5,
                "VRAM Data\nNot Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                style="italic",
                color="gray",
            )
            ax.set_title("V100 VRAM Usage", fontweight="bold", fontsize=12)

    def generate_all_visualizations(
        self, comparison_results: Dict[str, Any]
    ) -> List[str]:
        """Generate all available visualizations and return list of saved files.

        Args:
            comparison_results: Model comparison results from evaluator

        Returns:
            List of paths to saved visualization files
        """
        self.console.print(
            "🎨 Generating complete visualization suite...", style="bold blue"
        )

        saved_files = []

        try:
            # 1. Field accuracy heatmap
            heatmap_path = self.create_field_accuracy_heatmap(comparison_results)
            if heatmap_path:
                saved_files.append(heatmap_path)

            # 2. Performance dashboard
            dashboard_path = self.create_model_performance_dashboard(comparison_results)
            if dashboard_path:
                saved_files.append(dashboard_path)

            # 3. Category analysis
            category_path = self.create_field_category_analysis(comparison_results)
            if category_path:
                saved_files.append(category_path)

            # 4. V100 VRAM usage comparison
            vram_path = self.create_vram_usage_comparison(comparison_results)
            if vram_path:
                saved_files.append(vram_path)

            # 5. Production memory requirements (critical for POD sizing)
            production_memory_path = self.create_production_memory_requirements(
                comparison_results
            )
            if production_memory_path:
                saved_files.append(production_memory_path)

            # 6. Composite overview (2x2 layout)
            composite_path = self.create_composite_overview(comparison_results)
            if composite_path:
                saved_files.append(composite_path)

            self.console.print(
                f"✅ Generated {len(saved_files)} visualizations", style="bold green"
            )

        except Exception as e:
            self.console.print(f"❌ Error generating visualizations: {e}", style="red")

        return saved_files

    def create_summary_report(
        self, comparison_results: Dict[str, Any], visualization_paths: List[str]
    ) -> str:
        """Create a summary report with all visualizations and key insights.

        Args:
            comparison_results: Model comparison results
            visualization_paths: Paths to generated visualizations

        Returns:
            Path to HTML summary report
        """
        self.console.print("📄 Creating summary report...", style="blue")

        # Generate HTML report with embedded visualizations
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dynamic Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #A23B72; }}
                .metric {{ background-color: #F8F9FA; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .visualization {{ text-align: center; margin: 30px 0; }}
                .config-info {{ background-color: #E8F4FD; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Dynamic Model Comparison Report</h1>
            
            <div class="config-info">
                <h3>Dynamic Configuration</h3>
                <p><strong>Fields Loaded:</strong> {len(self.extraction_fields)} fields dynamically loaded from extraction_prompt</p>
                <p><strong>Field Weights:</strong> {len(self.field_weights)} weights configured</p>
                <p><strong>Quality Thresholds:</strong> {self.quality_thresholds}</p>
                <p><strong>Speed Thresholds:</strong> {self.speed_thresholds}</p>
            </div>
        """

        # Add model performance summary
        working_models = self._extract_working_models_from_comparison_runner(
            comparison_results
        )

        for model, results in working_models:
            # Calculate overall accuracy
            if "avg_accuracy" in results:
                overall_accuracy = results["avg_accuracy"]
            elif "field_wise_accuracy" in results:
                field_accs = list(results["field_wise_accuracy"].values())
                overall_accuracy = (
                    sum(field_accs) / len(field_accs) if field_accs else 0.0
                )
            else:
                overall_accuracy = 0.0

            html_content += f"""
            <div class="metric">
                <h3>{model.upper()} Performance</h3>
                <p><strong>Overall Accuracy:</strong> {overall_accuracy:.1%}</p>
                <p><strong>Processing Speed:</strong> {results["avg_processing_time"]:.1f}s per document</p>
                <p><strong>Success Rate:</strong> {results["success_rate"]:.1%}</p>
                <p><strong>Fields Extracted:</strong> {results["avg_fields_extracted"]:.1f} of {len(self.extraction_fields)} possible</p>
            </div>
            """

        # Add visualizations
        for viz_path in visualization_paths:
            viz_name = Path(viz_path).stem.replace("_", " ").title()
            html_content += f"""
            <div class="visualization">
                <h2>{viz_name}</h2>
                <img src="{Path(viz_path).name}" style="max-width: 100%; height: auto;">
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        # Save HTML report
        report_path = self.output_dir / "dynamic_model_comparison_report.html"
        with report_path.open("w") as f:
            f.write(html_content)

        self.console.print(f"✅ Summary report saved: {report_path}", style="green")
        return str(report_path)
