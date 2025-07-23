"""Main entry point for vision_processor CLI."""

import sys


def main():
    """Main CLI entry point with graceful error handling."""
    try:
        # Try to import and run the CLI
        from .cli.unified_cli import app

        app()
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print(
            "Please ensure all dependencies are installed and environment is activated:"
        )
        print("  conda activate unified_vision_processor")
        print("  pip install torch torchvision transformers")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
