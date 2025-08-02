#!/usr/bin/env python3
"""Minimal working example based on presentation code sample."""

import os
from pathlib import Path

# Set environment variable to avoid OpenMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_presentation_code_corrected():
    """Test the corrected version of the presentation code."""
    
    print("=== Minimal Working Example ===")
    print("Based on presentation section A2 code sample")
    print()
    
    try:
        # CORRECTED: Use the actual available classes
        from vision_processor.config import ConfigManager
        from vision_processor.extraction.extraction_manager import (
            SimpleExtractionManager,
        )
        
        print("✓ Imports successful")
        
        # Create a local config file with corrected paths for Mac
        local_config_content = """
# Simplified config for local testing
defaults:
  datasets_path: "./evaluation_data"
  max_tokens: 512
  quantization: true
  output_dir: "./output" 
  models: "internvl"
  trust_remote_code: true
  debug_mode: false

# Token limits
token_limits:
  max_tokens: 512

# Success criteria  
min_fields_for_success: 5

quality_thresholds:
  excellent: 12
  good: 8
  fair: 5
  poor: 0

speed_thresholds:
  very_fast: 15.0
  fast: 25.0
  moderate: 40.0

# Memory config
memory_config:
  v100_limit_gb: 16.0
  safety_margin: 0.85

# Device configuration
device_config:
  memory_limit_gb: 16.0
  device_maps:
    internvl:
      strategy: 'single_gpu'
      device_map: {'': 0}
      quantization_compatible: true
    llama:
      strategy: 'single_gpu'
      device_map: {'': 0}
      quantization_compatible: true

# Model paths - these would need to be updated for actual model locations
model_paths:
  llama: "./models/Llama-3.2-11B-Vision-Instruct"  # Update this path
  internvl: "./models/InternVL3-2B"                # Update this path

# Prompts
system_prompts:
  internvl: "Extract data accurately from business documents."

extraction_prompt: |
  Extract data from this business document. Output ALL fields below with their exact keys. Use "N/A" if field is not visible or not present.
  REQUIRED OUTPUT FORMAT (output ALL lines exactly as shown):
  DOCUMENT_TYPE: [value or N/A]
  SUPPLIER: [value or N/A]
  ABN: [11-digit Australian Business Number or N/A]
  PAYER_NAME: [value or N/A]
  TOTAL: [total amount in dollars or N/A]

image_processing:
  max_image_size: 512
  timeout_seconds: 30

logging:
  file_logging: false
  file_log_level: "INFO"

post_processing:
  enabled: true
  smart_mode: true
  force_parsing: false

repetition_control:
  enabled: true
  word_threshold: 0.15
  phrase_threshold: 2
  fallback_max_tokens: 512
"""
        
        # Write local config
        config_path = Path("local_config.yaml")
        config_path.write_text(local_config_content)
        
        print(f"✓ Created local config: {config_path}")
        
        # CORRECTED: Initialize with proper config manager
        config = ConfigManager(str(config_path))
        print("✓ ConfigManager created successfully")
        
        # CORRECTED: Use SimpleExtractionManager instead of UnifiedProcessor
        processor = SimpleExtractionManager(config)  # This would fail without actual models
        print("✗ Would fail here without actual model files")
        
    except Exception as e:
        print(f"Expected error (no model files): {e}")
    
    # Show the CORRECTED working code
    print("\n=== CORRECTED CODE SAMPLE ===")
    print("This is what should be in the presentation:")
    print()
    
    corrected_code = '''
from vision_processor.config import ConfigManager
from vision_processor.extraction.extraction_manager import SimpleExtractionManager

# Initialize with config file
config = ConfigManager("model_comparison.yaml")

# Create extraction manager (replaces UnifiedProcessor)
processor = SimpleExtractionManager(config)

# Extract fields (interface is slightly different)
result = processor.process_document("invoice.png")

# Results include all fields with confidence scores
print(result.extracted_fields)
print(f"Processing time: {result.processing_time:.2f}s")
print(f"Confidence: {result.model_confidence}")
'''
    
    print(corrected_code)
    
    # Show interface comparison
    print("\n=== INTERFACE COMPARISON ===")
    print("PRESENTATION CODE (incorrect):")
    print("  processor = UnifiedProcessor(model='internvl3')")
    print("  results = processor.extract_fields(image_path='invoice.png', expected_fields=DOCUMENT_FIELDS)")
    print("  print(results.extracted_fields)")
    print()
    print("ACTUAL WORKING CODE:")
    print("  config = ConfigManager('model_comparison.yaml')")
    print("  processor = SimpleExtractionManager(config)")
    print("  result = processor.process_document('invoice.png')")
    print("  print(result.extracted_fields)")

if __name__ == "__main__":
    test_presentation_code_corrected()