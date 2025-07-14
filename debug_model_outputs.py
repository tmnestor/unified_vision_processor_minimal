#!/usr/bin/env python3
"""Debug script to see what models actually output for receipt images."""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vision_processor.config.model_factory import ModelFactory
from vision_processor.config.simple_config import SimpleConfig


def test_model_output(model_type: str, test_image: str = "datasets/image14.png"):
    """Test what a model actually outputs for a given image."""
    print(f"\n{'=' * 60}")
    print(f"TESTING {model_type.upper()} MODEL")
    print(f"{'=' * 60}")

    try:
        # Setup model
        config = SimpleConfig()
        config.update_from_cli(model=model_type)
        model = ModelFactory.create_model(config)

        # Simple prompt
        simple_prompt = "<|image|>What text do you see in this image?"

        # Receipt extraction prompt
        extraction_prompt = """<|image|>Extract information from this receipt and return in KEY-VALUE format.

DATE: [date from receipt]
STORE: [store name]
TOTAL: [total amount]

Extract the visible text."""

        print("Testing with simple prompt:")
        print(f"Prompt: {simple_prompt}")
        response1 = model.process_image(test_image, simple_prompt)
        print(f"Response: {response1.raw_text[:200]}...")

        print("\nTesting with extraction prompt:")
        print(f"Prompt: {extraction_prompt}")
        response2 = model.process_image(test_image, extraction_prompt)
        print(f"Response: {response2.raw_text[:500]}...")

        return True

    except Exception as e:
        print(f"ERROR testing {model_type}: {e}")
        return False


def main():
    """Test both models to see their actual outputs."""
    print("🔍 DEBUGGING MODEL OUTPUTS")

    # Test both models
    test_model_output("internvl3")
    test_model_output("llama32_vision")

    print(f"\n{'=' * 60}")
    print("DEBUG COMPLETE - Check outputs above to fix extraction logic")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
