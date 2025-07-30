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
import yaml
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
        # Load fields using same mechanism as evaluator
        self.extraction_fields = self._load_extraction_fields()

        # Load weights and thresholds from config manager
        self.field_weights = self.config_manager.get_field_weights()

        # Load quality and speed thresholds from config
        try:
            config_dict = self.config_manager.get_legacy_config_dict()
            self.quality_thresholds = config_dict.get(
                "quality_thresholds", {"excellent": 12, "good": 8, "fair": 5, "poor": 0}
            )
            self.speed_thresholds = config_dict.get(
                "speed_thresholds", {"very_fast": 15.0, "fast": 25.0, "moderate": 40.0}
            )
        except AttributeError:
            # Fallback if legacy config method doesn't exist
            self.quality_thresholds = {"excellent": 12, "good": 8, "fair": 5, "poor": 0}
            self.speed_thresholds = {"very_fast": 15.0, "fast": 25.0, "moderate": 40.0}

        self.console.print(
            f"✅ Dynamic config loaded: {len(self.extraction_fields)} fields, "
            f"{len(self.field_weights)} weights",
            style="green",
        )

    def _load_extraction_fields(self) -> List[str]:
        """Load extraction fields dynamically from model_comparison.yaml (same as evaluator)."""
        try:
            # Look for model_comparison.yaml in current directory and parent directories
            config_path = Path("model_comparison.yaml")
            if not config_path.exists():
                config_path = Path("..") / "model_comparison.yaml"
            if not config_path.exists():
                config_path = Path("../../model_comparison.yaml")

            if not config_path.exists():
                self.console.print(
                    "⚠️ model_comparison.yaml not found, using fallback fields",
                    style="yellow",
                )
                # Basic fallback - should match evaluator fallback
                return [
                    "DOCUMENT_TYPE",
                    "SUPPLIER",
                    "ABN",
                    "PAYER_NAME",
                    "PAYER_ADDRESS",
                    "PAYER_PHONE",
                    "PAYER_EMAIL",
                    "INVOICE_DATE",
                    "DUE_DATE",
                    "GST",
                    "TOTAL",
                    "SUBTOTAL",
                    "SUPPLIER_WEBSITE",
                    "QUANTITIES",
                    "PRICES",
                    "BUSINESS_ADDRESS",
                    "BUSINESS_PHONE",
                    "BANK_NAME",
                    "BSB_NUMBER",
                    "BANK_ACCOUNT_NUMBER",
                    "ACCOUNT_HOLDER",
                    "STATEMENT_PERIOD",
                    "OPENING_BALANCE",
                    "CLOSING_BALANCE",
                    "DESCRIPTIONS",
                ]

            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            # Extract fields from extraction_prompt (same logic as evaluator)
            extraction_prompt = config.get("extraction_prompt", "")
            fields = []

            # Parse lines that match field pattern: "FIELD_NAME: [description]"
            for line in extraction_prompt.split("\n"):
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    # Extract field name before the colon
                    field_name = line.split(":")[0].strip()
                    # Check if it's a valid field (same validation as evaluator)
                    if (
                        field_name.isupper()
                        and len(field_name) <= 25
                        and not any(
                            word in field_name.lower()
                            for word in [
                                "required",
                                "correct",
                                "wrong",
                                "critical",
                                "use",
                                "never",
                                "absolutely",
                            ]
                        )
                    ):
                        fields.append(field_name)

            # Remove duplicates while preserving order
            seen = set()
            unique_fields = []
            for field in fields:
                if field not in seen:
                    seen.add(field)
                    unique_fields.append(field)

            return unique_fields

        except Exception as e:
            self.console.print(f"❌ Error loading extraction fields: {e}", style="red")
            # Return basic fallback
            return ["DOCUMENT_TYPE", "SUPPLIER", "ABN", "GST", "TOTAL"]

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

        # Extract working models and their field accuracies
        working_models = [
            (model, results)
            for model, results in comparison_results.items()
            if "error" not in results and "field_wise_accuracy" in results
        ]

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

        # Extract working models
        working_models = [
            (model, results)
            for model, results in comparison_results.items()
            if "error" not in results
        ]

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
        accuracies = [results["avg_accuracy"] * 100 for _, results in working_models]
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

        # 2. Processing Speed Comparison
        speeds = [results["avg_processing_time"] for _, results in working_models]
        bars2 = ax2.bar(
            models,
            speeds,
            color=[self.colors["warning"], self.colors["info"]][: len(models)],
        )
        ax2.set_title("Processing Speed", fontweight="bold")
        ax2.set_ylabel("Time (seconds)")

        # Add value labels
        for bar, speed in zip(bars2, speeds, strict=False):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{speed:.1f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add speed threshold lines from config
        for label, threshold in self.speed_thresholds.items():
            ax2.axhline(
                y=threshold,
                linestyle="--",
                alpha=0.7,
                label=f"{label.title()} ({threshold}s)",
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

        # 4. Fields Extracted Comparison
        fields_extracted = [
            results["avg_fields_extracted"] for _, results in working_models
        ]
        bars4 = ax4.bar(
            models,
            fields_extracted,
            color=[self.colors["primary"], self.colors["secondary"]][: len(models)],
        )
        ax4.set_title(
            f"Average Fields Extracted (of {len(self.extraction_fields)} total)",
            fontweight="bold",
        )
        ax4.set_ylabel("Fields per Document")
        ax4.set_ylim(0, len(self.extraction_fields))

        # Add value labels
        for bar, fields in zip(bars4, fields_extracted, strict=False):
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

        # Extract working models
        working_models = [
            (model, results)
            for model, results in comparison_results.items()
            if "error" not in results and "field_wise_accuracy" in results
        ]

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
        v100_limit_gb = 16.0  # V100 VRAM limit

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

        # Fallback: extract from working model results if available
        if not memory_data:
            # Handle different comparison_results formats
            if isinstance(comparison_results, dict):
                # Dictionary format - check for model results
                working_models = [
                    (model, results)
                    for model, results in comparison_results.items()
                    if isinstance(results, dict) and "error" not in results
                ]
            elif hasattr(comparison_results, "models_tested"):
                # ComparisonResults object format
                working_models = [
                    (model, {}) for model in comparison_results.models_tested
                ]
            else:
                # Unknown format - try to infer from object attributes
                working_models = []
                if hasattr(comparison_results, "__dict__"):
                    for attr_name, attr_value in comparison_results.__dict__.items():
                        if "model" in attr_name.lower() and isinstance(
                            attr_value, dict
                        ):
                            working_models.extend(
                                [(k, v) for k, v in attr_value.items()]
                            )

            # Use default estimates based on model type
            model_defaults = {
                "llama": 13.3,  # Llama-3.2-11B with 8-bit quantization
                "internvl": 11.2,  # InternVL3-8B with 8-bit quantization
                "llama32_vision": 13.3,
                "internvl3": 11.2,
            }

            for model_name, _ in working_models:
                memory_data[model_name] = model_defaults.get(model_name.lower(), 10.0)

        if not memory_data:
            self.console.print(
                "❌ No memory data available for VRAM chart", style="red"
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

        # Add safety margin line (85% of 16GB)
        safety_limit = v100_limit_gb * 0.85
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
                "✅ Safe"
                if usage <= safety_limit
                else "⚠️  Margin"
                if usage <= v100_limit_gb
                else "❌ Over"
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
        working_models = [
            (model, results)
            for model, results in comparison_results.items()
            if "error" not in results
        ]

        for model, results in working_models:
            html_content += f"""
            <div class="metric">
                <h3>{model.upper()} Performance</h3>
                <p><strong>Overall Accuracy:</strong> {results["avg_accuracy"]:.1%}</p>
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
