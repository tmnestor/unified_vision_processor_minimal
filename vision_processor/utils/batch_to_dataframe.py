"""Utility to convert batch_results.json to pandas DataFrame.

Converts batch processing results into a structured DataFrame with:
- One row per image
- One column per expected field (26 fields total)
- Missing values as None/NaN for better pandas analysis
"""

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..config.config_manager import ConfigManager


def batch_results_to_dataframe(
    batch_results_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    use_na_strings: bool = False,
) -> pd.DataFrame:
    """Convert batch_results.json to pandas DataFrame.

    Args:
        batch_results_path: Path to batch_results.json file
        config_path: Path to config YAML (optional, defaults to model_comparison.yaml)
        use_na_strings: If True, keep "N/A" strings; if False, convert to None/NaN

    Returns:
        pandas.DataFrame with columns [image, field1, field2, ...]

    Raises:
        FileNotFoundError: If batch_results.json doesn't exist
        ValueError: If batch_results.json is invalid
    """
    batch_path = Path(batch_results_path)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch results file not found: {batch_path}")

    # Load batch results
    try:
        with batch_path.open("r") as f:
            batch_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {batch_path}: {e}") from e

    # Get expected fields from config
    if config_path:
        config = ConfigManager(str(config_path))
    else:
        # Try to find config in parent directories
        current_dir = batch_path.parent
        config_file = None
        for _ in range(3):  # Search up to 3 levels up
            test_path = current_dir / "model_comparison.yaml"
            if test_path.exists():
                config_file = test_path
                break
            current_dir = current_dir.parent

        if not config_file:
            raise FileNotFoundError(
                "Could not find model_comparison.yaml. Please specify config_path."
            )
        config = ConfigManager(str(config_file))

    expected_fields = config.get_expected_fields()

    # Prepare DataFrame columns: image + all expected fields
    columns = ["image"] + expected_fields
    rows = []

    # Handle different JSON structures
    if isinstance(batch_data, list):
        # Direct list of results
        results = batch_data
    elif isinstance(batch_data, dict) and "results" in batch_data:
        # Wrapped in "results" key
        results = batch_data["results"]
    else:
        # Fallback: try to extract from top level
        results = batch_data.get("results", [])
    
    if not results:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=columns)

    for result in results:
        # Handle case where result might not be a dictionary
        if not isinstance(result, dict):
            print(f"WARNING: Expected dict but got {type(result)}: {result}")
            continue
            
        # Get image name (could be "filename" or "image")
        image_name = result.get("filename", result.get("image", "unknown"))
        if isinstance(image_name, str) and "/" in image_name:
            # Extract just filename from path
            image_name = Path(image_name).name

        # Get extracted fields (they ARE nested under "extracted_fields")
        extracted_fields = result.get("extracted_fields", {})

        # Build row starting with image name
        row = {"image": image_name}

        # Add each expected field from extracted_fields
        for field in expected_fields:
            value = extracted_fields.get(field, "N/A")

            # Convert "N/A" to None/NaN unless user wants to keep strings
            if not use_na_strings and value == "N/A":
                value = None

            row[field] = value

        rows.append(row)

    # Create DataFrame
    batch_dataframe = pd.DataFrame(rows, columns=columns)

    return batch_dataframe


def save_dataframe_to_csv(
    batch_results_path: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    use_na_strings: bool = False,
) -> Path:
    """Convert batch_results.json to CSV file.

    Args:
        batch_results_path: Path to batch_results.json file
        output_csv_path: Output CSV path (optional, auto-generated if None)
        config_path: Path to config YAML (optional)
        use_na_strings: If True, keep "N/A" strings; if False, convert to None/NaN

    Returns:
        Path to created CSV file
    """
    # Generate output path if not provided
    if output_csv_path is None:
        batch_path = Path(batch_results_path)
        
        # Detect model name from batch results for filename
        model_name = _detect_model_name_from_batch_file(batch_results_path)
        
        # Include model name in filename
        if model_name:
            filename = f"{batch_path.stem}_{model_name}_dataframe.csv"
        else:
            filename = f"{batch_path.stem}_dataframe.csv"
        
        output_csv_path = batch_path.parent / filename

    # Convert to DataFrame
    batch_dataframe = batch_results_to_dataframe(
        batch_results_path=batch_results_path,
        config_path=config_path,
        use_na_strings=use_na_strings,
    )

    # Save to CSV
    output_path = Path(output_csv_path)
    batch_dataframe.to_csv(output_path, index=False)

    print(f"✅ Converted {len(batch_dataframe)} images to CSV: {output_path}")
    print(f"📊 DataFrame shape: {batch_dataframe.shape} (rows, columns)")
    print(f"📋 Columns: image + {len(batch_dataframe.columns) - 1} fields")

    return output_path


def _detect_model_name_from_batch_file(batch_results_path: Union[str, Path]) -> Optional[str]:
    """Detect model name from batch results file.
    
    Args:
        batch_results_path: Path to batch results JSON file
        
    Returns:
        Model name if detected, None otherwise
    """
    try:
        batch_path = Path(batch_results_path)
        if not batch_path.exists():
            print(f"DEBUG: Batch file not found: {batch_path}")
            return None
            
        with batch_path.open("r") as f:
            batch_data = json.load(f)
        
        model_name = _detect_model_name_from_batch(batch_data)
        print(f"DEBUG: Detected model name: {model_name}")
        return model_name
    except (json.JSONDecodeError, IOError) as e:
        print(f"DEBUG: Error reading batch file: {e}")
        return None


def _detect_model_name_from_batch(batch_data) -> Optional[str]:
    """Detect model name from batch results data.
    
    Args:
        batch_data: Parsed batch results JSON data
        
    Returns:
        Model name if detected, None otherwise
    """
    print(f"DEBUG: Batch data type: {type(batch_data)}")
    if isinstance(batch_data, list):
        print(f"DEBUG: List length: {len(batch_data)}")
        if batch_data:
            print(f"DEBUG: First result keys: {list(batch_data[0].keys()) if isinstance(batch_data[0], dict) else 'Not a dict'}")
    elif isinstance(batch_data, dict):
        print(f"DEBUG: Dict keys: {list(batch_data.keys())}")
    
    # Handle different batch result formats
    if isinstance(batch_data, list) and batch_data:
        # Direct list format - check first result
        first_result = batch_data[0]
        if isinstance(first_result, dict) and "model" in first_result:
            return first_result["model"]
    elif isinstance(batch_data, dict):
        # Check if there's a results key
        if "results" in batch_data and batch_data["results"]:
            first_result = batch_data["results"][0]
            if isinstance(first_result, dict) and "model" in first_result:
                return first_result["model"]
        # Check if there's a model key at the top level
        elif "model" in batch_data:
            return batch_data["model"]
    
    return None


def print_dataframe_info(batch_dataframe: pd.DataFrame) -> None:
    """Print useful information about the DataFrame.

    Args:
        batch_dataframe: DataFrame to analyze
    """
    print("\n📊 DataFrame Info:")
    print(f"Shape: {batch_dataframe.shape} (rows, columns)")
    print(f"Images: {len(batch_dataframe)}")
    print(f"Fields: {len(batch_dataframe.columns) - 1}")  # Subtract 1 for image column

    # Show missing value counts for each field
    print("\n📋 Missing Values per Field:")
    missing_counts = batch_dataframe.isna().sum()
    field_columns = [col for col in batch_dataframe.columns if col != "image"]

    for field in field_columns:
        missing = missing_counts[field]
        present = len(batch_dataframe) - missing
        percentage = (
            (present / len(batch_dataframe)) * 100 if len(batch_dataframe) > 0 else 0
        )
        print(f"  {field}: {present}/{len(batch_dataframe)} ({percentage:.1f}%)")

    # Show overall extraction statistics
    total_possible = len(batch_dataframe) * len(field_columns)
    total_extracted = batch_dataframe[field_columns].count().sum()
    overall_percentage = (
        (total_extracted / total_possible) * 100 if total_possible > 0 else 0
    )

    print("\n🎯 Overall Extraction:")
    print(
        f"  Total extracted: {total_extracted}/{total_possible} ({overall_percentage:.1f}%)"
    )


if __name__ == "__main__":
    """CLI usage example."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert batch_results.json to pandas DataFrame"
    )
    parser.add_argument("batch_results", help="Path to batch_results.json file")
    parser.add_argument("--output", "-o", help="Output CSV path (optional)")
    parser.add_argument("--config", "-c", help="Config YAML path (optional)")
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep 'N/A' strings instead of converting to NaN",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Show DataFrame info without saving CSV",
    )

    args = parser.parse_args()

    try:
        # Convert to DataFrame
        batch_dataframe = batch_results_to_dataframe(
            batch_results_path=args.batch_results,
            config_path=args.config,
            use_na_strings=args.keep_na,
        )

        # Show info
        print_dataframe_info(batch_dataframe)

        # Save CSV unless info-only mode
        if not args.info_only:
            csv_path = save_dataframe_to_csv(
                batch_results_path=args.batch_results,
                output_csv_path=args.output,
                config_path=args.config,
                use_na_strings=args.keep_na,
            )
            print(f"\n💾 Saved to: {csv_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
