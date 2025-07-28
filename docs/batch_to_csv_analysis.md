# Batch Results to DataFrame Converter

## Overview

The `batch_to_csv_cli.py` script is a critical data analysis tool that converts batch vision processing results into structured pandas DataFrames. This enables easy statistical analysis, visualization, and reporting of document field extraction performance across large datasets.

## Purpose

After running batch document processing with the vision processor, you get a `batch_results.json` file containing raw extraction results. This script transforms that data into a clean, analysis-ready CSV format with one row per image and one column per field.

## Architecture

### Core Components

```
vision_processor/cli/batch_to_csv_cli.py
├── convert command     # Main conversion functionality
├── analyze command     # Analysis without CSV export
└── utils integration   # Leverages batch_to_dataframe.py
```

### Data Flow

```
batch_results.json → JSON parsing → DataFrame creation → CSV export
                  ↓                 ↓                   ↓
              Field extraction   Column mapping    Analysis-ready data
```

## CLI Interface

### Commands

#### 1. Convert Command
```bash
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml
```

**Purpose**: Converts batch results to DataFrame and saves as CSV

**Parameters**:
- `batch_file`: Path to batch_results.json (resolved to output directory if relative)
- `--yaml-file`: Configuration file (required for field schema)
- `--output`: Custom CSV output path (optional)
- `--keep-na`: Keep "N/A" strings instead of converting to NaN (optional)
- `--info`: Show DataFrame info without saving CSV (optional)

#### 2. Analyze Command
```bash
python -m vision_processor.cli.batch_to_csv_cli analyze batch_results.json --yaml-file model_comparison.yaml
```

**Purpose**: Analyzes batch results and shows statistics without creating CSV

**Features**:
- DataFrame shape and structure info
- Field-by-field extraction success rates
- Overall extraction completeness percentage
- Sample data preview

## Technical Implementation

### JSON Structure Handling

The script handles the actual batch_results.json structure:

```json
[
  {
    "filename": "image01.png",
    "extracted_fields": {
      "DOCUMENT_TYPE": "TAX INVOICE",
      "SUPPLIER": "JB Hi-Fi",
      "ABN": "17 148 660 920",
      // ... 26 total fields
    },
    "processing_time": 32.598,
    "model_confidence": 0.85,
    "extraction_method": "clean_with_repetition"
  }
  // ... more results
]
```

### Path Resolution

The script uses intelligent path resolution:

1. **Input Files**: If `batch_results.json` has no path separator, looks in configured output directory
2. **Output Files**: CSV files are saved to the same directory as the source JSON
3. **Config Files**: Uses specified YAML file for field schema

### DataFrame Structure

**Output DataFrame Format**:
```
Columns: [image, DOCUMENT_TYPE, SUPPLIER, ABN, PAYER_NAME, ...]
Rows: One per processed image
Values: Extracted field values or NaN for missing data
```

**Example**:
| image | DOCUMENT_TYPE | SUPPLIER | ABN | PAYER_NAME | ... |
|-------|---------------|----------|-----|------------|-----|
| image01.png | TAX INVOICE | JB Hi-Fi | 17 148 660 920 | David Miller | ... |
| image02.png | RECEIPT | Woolworths | NaN | John Smith | ... |

## Error Handling

### Robust Configuration Loading
```python
try:
    config_manager = ConfigManager(config_file)
    output_dir = config_manager.output_dir
    # Use config-based path resolution
except Exception as e:
    # Fallback to current directory
    console.print(f"⚠️  Config loading failed ({e}), using current directory")
```

### JSON Structure Flexibility
The script handles multiple JSON formats:
- Direct list of results (current format)
- Wrapped in "results" key (alternative format)
- Mixed data types with validation

### Missing Data Handling
- Converts "N/A" strings to pandas NaN by default
- Option to preserve "N/A" strings with `--keep-na`
- Handles missing fields gracefully

## Analytics Features

### Field-Level Statistics
```
📋 Missing Values per Field:
  DOCUMENT_TYPE: 24/24 (100.0%)
  SUPPLIER: 17/24 (70.8%)
  ABN: 19/24 (79.2%)
  PAYER_EMAIL: 1/24 (4.2%)
```

### Overall Performance Metrics
```
🎯 Overall Extraction:
  Total extracted: 310/624 (49.7%)
```

### Sample Data Preview
Shows first 3 rows with first 5 fields to verify data structure.

## Integration with Vision Processor

### Workflow Integration
```bash
# Step 1: Run batch processing
python -m vision_processor.cli.extract_cli batch ./datasets/ --model llama

# Step 2: Convert to DataFrame
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml

# Step 3: Analyze with pandas/Excel
python -c "import pandas as pd; df = pd.read_csv('batch_results_dataframe.csv'); print(df.describe())"
```

### Configuration Dependency
The script requires the same YAML configuration file used for processing to:
- Extract the expected field schema (26 fields)
- Resolve paths correctly
- Maintain consistency across the pipeline

## Use Cases

### 1. Model Performance Analysis
```python
import pandas as pd
df = pd.read_csv('batch_results_dataframe.csv')

# Calculate success rates per field
success_rates = df.notna().mean() * 100
print(success_rates.sort_values(ascending=False))
```

### 2. Data Quality Assessment
```python
# Find images with low extraction rates
extraction_counts = df.notna().sum(axis=1)
poor_extractions = df[extraction_counts < 10]
print(f"Images with <10 fields: {len(poor_extractions)}")
```

### 3. Field Difficulty Analysis
```python
# Identify problematic fields
field_success = df.notna().mean()
difficult_fields = field_success[field_success < 0.3]
print("Fields extracted <30% of the time:", difficult_fields.index.tolist())
```

## Production Considerations

### Memory Efficiency
- Processes large batch files without loading everything into memory
- Converts data types appropriately (strings → NaN for missing values)
- Optimized pandas DataFrame creation

### KFP Compatibility
- Rich console output degrades gracefully in non-TTY environments
- Standard logging for critical operations
- Exit codes for pipeline automation

### Error Recovery
- Continues processing even if individual records have issues
- Detailed error messages with remediation suggestions
- Fallback behaviors for missing configurations

## Output Files

### CSV Structure
- **Location**: Same directory as input JSON (usually output directory)
- **Naming**: `{input_filename}_dataframe.csv`
- **Format**: Standard CSV with headers, suitable for Excel/pandas
- **Encoding**: UTF-8 with proper handling of special characters

### Analysis Summary
Printed to console during conversion:
- DataFrame dimensions
- Field extraction success rates
- Overall completion percentage
- File locations

## Best Practices

### 1. Always specify --yaml-file
```bash
# Good
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml

# Avoid - may fail to find config
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json
```

### 2. Use analyze command for exploration
```bash
# Quick overview without creating files
python -m vision_processor.cli.batch_to_csv_cli analyze batch_results.json --yaml-file model_comparison.yaml
```

### 3. Check data quality first
Always review the extraction statistics before proceeding with analysis.

## Troubleshooting

### Common Issues

**"Could not find model_comparison.yaml"**
- Solution: Specify full path to YAML file or ensure it's in current directory

**"File not found: batch_results.json"**
- Solution: Check that batch processing completed successfully
- Verify you're in the correct directory or use full path

**"'list' object has no attribute 'get'"**
- Cause: Unexpected JSON structure
- Solution: Check the JSON format in batch_results.json

### Debug Mode
Add verbose output by examining the script's console messages for detailed path resolution and data loading information.

## Future Enhancements

### Potential Improvements
1. **Multiple format support**: Excel, Parquet, JSON output
2. **Advanced analytics**: Built-in visualization and reporting
3. **Incremental processing**: Append new results to existing DataFrames
4. **Schema validation**: Verify field consistency across batches
5. **Performance metrics**: Processing time analysis per image

### Extension Points
The modular design allows easy extension for:
- Custom data transformations
- Additional output formats  
- Integration with visualization tools
- Automated report generation

## Conclusion

The `batch_to_csv_cli.py` script is an essential bridge between raw vision processing results and structured data analysis. It transforms complex nested JSON into clean, analysis-ready DataFrames while maintaining data integrity and providing comprehensive statistics about extraction performance.

This tool enables data scientists and analysts to quickly assess model performance, identify problematic fields or images, and generate insights from large-scale document processing operations.