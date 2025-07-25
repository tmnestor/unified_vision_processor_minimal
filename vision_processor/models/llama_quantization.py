"""Llama-3.2-Vision Quantization Management

Handles all quantization configuration and memory estimation logic.
"""

import logging
from typing import Optional

import torch

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

logger = logging.getLogger(__name__)


class LlamaQuantizationManager:
    """Manages quantization configuration for Llama-3.2-Vision model."""

    def __init__(self, config, enable_quantization: bool = True, **kwargs):
        """Initialize quantization manager.
        
        Args:
            config: Model configuration object
            enable_quantization: Whether to enable quantization
            **kwargs: Additional arguments (e.g., aggressive_quantization)
        """
        self.config = config
        self.enable_quantization = enable_quantization
        self.kwargs = kwargs

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if enabled."""
        if not self.enable_quantization:
            return None

        if BitsAndBytesConfig is None:
            logger.warning("BitsAndBytesConfig not available - falling back to FP16")
            return None

        try:
            # Check if aggressive quantization is requested for V100 safety
            use_4bit = self.kwargs.get("aggressive_quantization", False)

            if use_4bit:
                # Use 4-bit quantization for 11B model V100 deployment
                logger.info(
                    "🔧 Using 4-bit quantization for aggressive memory reduction"
                )
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                # Use int8 quantization for standard V100 deployment
                logger.info("🔧 Using 8-bit quantization for V100 deployment")
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                    llm_int8_threshold=6.0,
                )
        except Exception as e:
            logger.warning(f"Failed to create quantization config: {e}")
            return None

    def estimate_memory_usage(
        self, quantization_config: Optional[BitsAndBytesConfig]
    ) -> float:
        """Estimate model memory usage in GB for V100 validation.
        
        Args:
            quantization_config: Quantization configuration
            
        Returns:
            Estimated memory usage in GB
        """
        # Base model size: Llama-3.2-11B ≈ 11B parameters
        base_params = 11_000_000_000  # 11B parameters

        if quantization_config:
            if (
                hasattr(quantization_config, "load_in_4bit")
                and quantization_config.load_in_4bit
            ):
                # 4-bit quantization: ~0.5 bytes per parameter
                memory_gb = (base_params * 0.5) / (1024**3)
                overhead = 1.5  # Quantization overhead
                return memory_gb * overhead
            elif (
                hasattr(quantization_config, "load_in_8bit")
                and quantization_config.load_in_8bit
            ):
                # 8-bit quantization: ~1 byte per parameter
                memory_gb = (base_params * 1.0) / (1024**3)
                overhead = 1.3  # Quantization overhead
                return memory_gb * overhead

        # FP16: 2 bytes per parameter
        memory_gb = (base_params * 2.0) / (1024**3)
        overhead = 1.2  # Model overhead
        return memory_gb * overhead

    def validate_v100_compliance(self, estimated_memory_gb: float) -> None:
        """Validate that estimated memory usage complies with V100 limits.
        
        Args:
            estimated_memory_gb: Estimated memory usage in GB
            
        Raises:
            RuntimeError: If memory usage exceeds V100 limits
        """
        # Get memory config from YAML (single source of truth)
        memory_config = getattr(self.config, "yaml_config", {}).get("memory_config", {})
        v100_limit_gb = memory_config.get("v100_limit_gb", 16.0)
        safety_margin = memory_config.get("safety_margin", 0.85)
        effective_limit = v100_limit_gb * safety_margin

        if estimated_memory_gb > effective_limit:
            logger.error("❌ V100 COMPLIANCE FAILURE:")
            logger.error(f"   Estimated memory: {estimated_memory_gb:.1f}GB")
            logger.error(f"   V100 safe limit: {effective_limit:.1f}GB (85% of 16GB)")
            logger.error(f"   Excess: {estimated_memory_gb - effective_limit:.1f}GB")
            logger.error("")
            logger.error("💡 SOLUTIONS:")
            if not self.enable_quantization:
                logger.error("   1. Enable quantization (set quantization=true)")
            else:
                use_4bit = self.kwargs.get("aggressive_quantization", False)
                if not use_4bit:
                    logger.error("   1. Enable aggressive 4-bit quantization")
                    logger.error(
                        "   2. Use smaller model variant (e.g., 7B instead of 11B)"
                    )
                else:
                    logger.error(
                        "   1. Use smaller model variant (e.g., 7B instead of 11B)"
                    )

            raise RuntimeError(
                f"Model memory ({estimated_memory_gb:.1f}GB) exceeds V100 safe limit ({effective_limit:.1f}GB)"
            )
        else:
            safety_percent = (
                (effective_limit - estimated_memory_gb) / effective_limit * 100
            )
            logger.info(
                f"✅ V100 Compliance: {estimated_memory_gb:.1f}GB / {v100_limit_gb:.1f}GB ({safety_percent:.1f}% safety margin)"
            )

    def apply_quantization(self, model) -> None:
        """Apply quantization to model (handled during loading).
        
        Args:
            model: The loaded model instance
        """
        if not self.enable_quantization:
            logger.info("Quantization disabled")
            return

        logger.info("Quantization applied during model loading")


