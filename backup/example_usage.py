#!/usr/bin/env python3
"""Example usage of the simplified vision processor."""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set required environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def example_basic_usage():
    """Example of basic document processing."""
    print("📄 Example: Basic Document Processing")
    print("-" * 40)

    try:
        from vision_processor.config.simple_config import SimpleConfig

        # Load configuration from .env
        config = SimpleConfig()
        print("Configuration loaded from .env file")

        # You would normally process a real image like this:
        # from vision_processor.extraction.simple_extraction_manager import SimpleExtractionManager
        # manager = SimpleExtractionManager(config)
        # result = manager.process_document("path/to/receipt.jpg")

        print("✅ Basic usage example complete")
        print("   To process a real document:")
        print("   python -m vision_processor.cli.simple_extract_cli extract receipt.jpg")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_configuration():
    """Example of configuration management."""
    print("\n⚙️  Example: Configuration Management")
    print("-" * 40)

    try:
        from vision_processor.config.simple_config import SimpleConfig

        # Load default configuration
        config = SimpleConfig()

        print("Default configuration:")
        config.print_configuration()

        # Show validation
        print(f"\nConfiguration valid: {config.validate()}")

        # Show model config dict
        model_config = config.get_model_config()
        print(f"\nModel config keys: {list(model_config.keys())}")

        print("✅ Configuration example complete")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_key_schema():
    """Example of key schema usage."""
    print("\n🔑 Example: Key Schema Management")
    print("-" * 40)

    try:
        import yaml

        from vision_processor.extraction.universal_key_value_parser import UniversalKeyValueParser

        # Load key schema from prompts.yaml
        prompts_path = project_root / "vision_processor" / "config" / "prompts.yaml"
        with prompts_path.open("r") as f:
            prompts_data = yaml.safe_load(f)

        key_schema = prompts_data["key_schema"]

        print(f"Required keys: {key_schema['required_keys']}")
        print(f"Optional keys (first 10): {key_schema['optional_keys'][:10]}...")
        print(f"Key patterns: {key_schema['key_patterns']}")

        # Create parser
        parser = UniversalKeyValueParser(key_schema)

        # Example response parsing
        example_response = """
        DATE: 17/10/2020
        STORE: WOOLWORTHS
        ABN: 88 000 014 675
        GST: 27.84
        TOTAL: 306.28
        ITEMS: Bread White | Milk 2L | Eggs Free Range 12pk
        RECEIPT_NUMBER: 745392
        PAYMENT_METHOD: AMEX
        """

        extracted = parser.parse(example_response)
        print(f"\nParsed {len(extracted)} fields from example response:")
        for key, value in extracted.items():
            print(f"  {key}: {value}")

        # Validate
        validation = parser.validate_extraction(extracted)
        print(f"\nValidation result: {validation['is_valid']}")
        if validation["missing_required"]:
            print(f"Missing required: {validation['missing_required']}")

        print("✅ Key schema example complete")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_cli_usage():
    """Example of CLI usage."""
    print("\n🖥️  Example: CLI Usage")
    print("-" * 40)

    try:
        print("Available CLI commands:")
        print("")
        print("1. Extract single document:")
        print("   python -m vision_processor.cli.simple_extract_cli extract receipt.jpg")
        print("")
        print("2. Extract with model override:")
        print(
            "   python -m vision_processor.cli.simple_extract_cli extract receipt.jpg --model llama32_vision"
        )
        print("")
        print("3. Extract with output format override:")
        print(
            "   python -m vision_processor.cli.simple_extract_cli extract receipt.jpg --output-format json"
        )
        print("")
        print("4. Compare models:")
        print("   python -m vision_processor.cli.simple_extract_cli compare receipt.jpg")
        print("")
        print("5. Show configuration:")
        print("   python -m vision_processor.cli.simple_extract_cli config-info")
        print("")
        print("6. Batch processing:")
        print(
            "   python -m vision_processor.cli.simple_extract_cli batch ./images/ --output-dir ./results/"
        )
        print("")

        print("✅ CLI usage examples complete")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_env_configuration():
    """Example of .env configuration options."""
    print("\n🔧 Example: .env Configuration Options")
    print("-" * 40)

    try:
        env_path = project_root / ".env"

        print(f".env file location: {env_path}")
        print(f"File exists: {env_path.exists()}")

        if env_path.exists():
            print(f"File size: {env_path.stat().st_size} bytes")

            # Show first few lines
            with env_path.open("r") as f:
                lines = f.readlines()[:10]

            print("\nFirst 10 lines of .env:")
            for i, line in enumerate(lines, 1):
                print(f"  {i:2}: {line.rstrip()}")

        print("\nKey configuration options:")
        print("  VISION_MODEL_TYPE: internvl3 | llama32_vision")
        print("  VISION_MODEL_PATH: /path/to/model")
        print("  VISION_DEVICE_CONFIG: auto | cuda:0 | cpu")
        print("  VISION_OUTPUT_FORMAT: table | json | yaml")
        print("  VISION_ENABLE_QUANTIZATION: true | false")

        print("✅ .env configuration example complete")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples."""
    print("🚀 Simplified Vision Processor - Usage Examples")
    print("=" * 60)

    examples = [
        example_basic_usage,
        example_configuration,
        example_key_schema,
        example_cli_usage,
        example_env_configuration,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example failed: {e}")

    print("\n" + "=" * 60)
    print("📚 Documentation and Next Steps")
    print("=" * 60)
    print("1. Edit .env file to configure your model paths")
    print("2. Run test_simple_extraction.py to verify setup")
    print("3. Use the CLI to process documents")
    print("4. Modify prompts.yaml to customize extraction keys")
    print("")
    print("For help with CLI:")
    print("  python -m vision_processor.cli.simple_extract_cli --help")


if __name__ == "__main__":
    main()
