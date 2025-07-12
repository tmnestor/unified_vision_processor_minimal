"""Extraction Package

Implements the unified 7-step processing pipeline combining Llama architecture
with InternVL enhancements for robust document processing.
"""

# Import from new organized modules
from ..classification import ClassificationResult, DocumentClassifier, DocumentType
from ..confidence import ComplianceResult, ConfidenceManager, ConfidenceResult
from .awk_extractor import AWKExtractor, ExtractionPattern, FieldType
from .hybrid_extraction_manager import (
    ProcessingResult,
    ProcessingStage,
    QualityGrade,
    UnifiedExtractionManager,
)
from .pipeline_components import (
    ATOComplianceHandler,
    DocumentHandler,
    EnhancedKeyValueParser,
    HighlightDetector,
    PromptManager,
)

__all__ = [
    "ATOComplianceHandler",
    "AWKExtractor",
    "ClassificationResult",
    "ComplianceResult",
    "ConfidenceManager",
    "ConfidenceResult",
    "DocumentClassifier",
    "DocumentHandler",
    "DocumentType",
    "EnhancedKeyValueParser",
    "ExtractionPattern",
    "FieldType",
    "HighlightDetector",
    "ProcessingResult",
    "ProcessingStage",
    "PromptManager",
    "QualityGrade",
    "UnifiedExtractionManager",
]
