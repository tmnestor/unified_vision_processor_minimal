#!/usr/bin/env python3
"""Test individual model classification for debugging.

Quick test to verify each model works independently before running comparison.
"""

import sys
from pathlib import Path

from vision_processor.config.model_factory import ModelFactory
from vision_processor.config.simple_config import SimpleConfig


def test_internvl_classification():
    """Test InternVL3 classification only."""
    print("=== TESTING INTERNVL3 CLASSIFICATION ===")

    config = SimpleConfig()
    config.model_type = "internvl3"
    config.model_path = "/home/jovyan/nfs_share/models/InternVL3-8B"

    try:
        model = ModelFactory.create_model(config)

        # Test with image14 (known working receipt)
        test_image = "datasets/image14.png"
        if not Path(test_image).exists():
            print(f"Error: {test_image} not found")
            return False

        print(f"Classifying: {test_image}")
        result = model.classify_document(test_image)

        print(f"Document Type: {result['document_type']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Is Business Document: {result['is_business_document']}")
        print(f"Response: {result['classification_response'][:200]}...")

        return True

    except Exception as e:
        print(f"InternVL3 Error: {e}")
        return False


def test_llama_classification():
    """Test Llama-3.2-Vision classification only."""
    print("=== TESTING LLAMA-3.2-VISION CLASSIFICATION ===")

    config = SimpleConfig()
    config.model_type = "llama32_vision"
    config.model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"

    try:
        model = ModelFactory.create_model(config)

        # Test with image14 (known working receipt)
        test_image = "datasets/image14.png"
        if not Path(test_image).exists():
            print(f"Error: {test_image} not found")
            return False

        print(f"Classifying: {test_image}")
        result = model.classify_document(test_image)

        print(f"Document Type: {result['document_type']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Is Business Document: {result['is_business_document']}")
        print(f"Response: {result['classification_response'][:200]}...")

        return True

    except Exception as e:
        print(f"Llama-3.2-Vision Error: {e}")
        return False


def main():
    """Test individual models."""
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type == "internvl":
            test_internvl_classification()
        elif model_type == "llama":
            test_llama_classification()
        else:
            print("Usage: python test_individual_classification.py [internvl|llama]")
    else:
        print("Testing both models individually...")
        print()

        internvl_ok = test_internvl_classification()
        print("\n" + "=" * 50 + "\n")
        llama_ok = test_llama_classification()

        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"InternVL3: {'✓ OK' if internvl_ok else '✗ FAILED'}")
        print(f"Llama-3.2: {'✓ OK' if llama_ok else '✗ FAILED'}")

        if internvl_ok and llama_ok:
            print("\n✓ Both models ready for comparison!")
        else:
            print("\n⚠ Fix individual model issues before running comparison")


if __name__ == "__main__":
    main()