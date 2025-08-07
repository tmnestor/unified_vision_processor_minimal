#!/usr/bin/env python
# coding: utf-8

"""
Environment Setup and Model Loading for InternVL3 Evaluation

Purpose:
- Import all required libraries for InternVL3-2B vision-language model and evaluation processing
- Load the InternVL3-2B model with compatibility fixes for stable inference
- Initialize model with proper dtype and settings for evaluation against ground truth
- Define global configuration variables for evaluation data paths and ground truth

Key Components (Compatibility Optimized):
- torch.bfloat16: Recommended precision for optimal performance
- use_flash_attn=False: Disabled for compatibility (fixes dtype mismatch errors)
- low_cpu_mem_usage=True: Optimize CPU memory during loading
- trust_remote_code=True: Allow loading custom model code from HuggingFace
- .eval().cuda(): Set model to evaluation mode and move to GPU

Evaluation Configuration:
- data_dir: Path to evaluation_data directory with 20 test images
- ground_truth_path: Path to evaluation_ground_truth.csv with correct answers
- output_dir: Directory for saving evaluation results and performance reports

Evaluation Libraries:
- pandas: For CSV handling and ground truth comparison
- pathlib: For robust file path handling
- pathlib.Path.glob: For image file discovery
- re: For sophisticated field comparison (numeric, date, string matching)
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T

# Check transformers version (should be >=4.37.2)
import transformers
from PIL import Image
from transformers import AutoModel, AutoTokenizer

print(f"🔍 Transformers version: {transformers.__version__}")

# Global configuration variables for evaluation
data_dir = "/home/jovyan/nfs_share/tod/evaluation_data"  # 20 test images
ground_truth_path = "/home/jovyan/nfs_share/tod/unified_vision_processor_minimal/evaluation_ground_truth.csv"  # Ground truth CSV
model_path = "/home/jovyan/nfs_share/models/InternVL3-2B"
output_dir = "/home/jovyan/nfs_share/tod/output"

# data_dir = "/efs/share/PoC_data/evaluation_data"  # 20 test images
# ground_truth_path = "/efs/share/PoC_data/evaluation_ground_truth.csv"  # Ground truth CSV
# model_path = "/efs/share/PTM/InternVL3-2B"
# output_dir = "/efs/share/PoC_data/output"

print("🎯 EVALUATION MODE ENABLED")
print(f"📁 Evaluation data directory: {data_dir}")
print(f"📊 Ground truth file: {ground_truth_path}")
print(f"📁 Output directory: {output_dir}")
print(f"🔧 Loading InternVL3-2B model with compatibility fixes from: {model_path}")

# Validate evaluation setup
evaluation_data_path = Path(data_dir)
gt_path = Path(ground_truth_path)

if not evaluation_data_path.exists():
    print(f"❌ ERROR: Evaluation data directory not found: {data_dir}")
    print("💡 Please ensure evaluation_data/ directory exists")
else:
    test_images = list(evaluation_data_path.glob("synthetic_invoice_*.png"))
    print(f"✅ Found {len(test_images)} test images in evaluation_data/")

if not gt_path.exists():
    print(f"❌ ERROR: Ground truth file not found: {ground_truth_path}")
    print("💡 Please ensure evaluation_ground_truth.csv exists")
else:
    print(f"✅ Ground truth file found: {gt_path.name}")

# Load model with compatibility settings (use_flash_attn=False to fix dtype errors)
model = (
    AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Official recommendation: use bfloat16
        low_cpu_mem_usage=True,  # Optimize CPU memory during loading
        use_flash_attn=False,  # FIXED: Disabled for compatibility (prevents dtype mismatch)
        trust_remote_code=True,  # Allow custom model code execution
    )
    .eval()
    .cuda()
)  # Set to evaluation mode and move to GPU

# Load tokenizer with official settings
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,  # Allow custom tokenizer code
    use_fast=False,  # Use slower but more reliable tokenizer for structured tasks
)

print("✅ Model and tokenizer loaded successfully with compatibility fixes")
print("📊 Evaluation libraries imported: pandas, glob, datetime, re, json")
print("🎯 Ready for rigorous InternVL3 evaluation against ground truth dataset")


"""
Official InternVL3 Dynamic Image Processing Pipeline

Purpose:
- Implement official InternVL3 dynamic image preprocessing following documentation
- Support dynamic tiling with proper dtype consistency
- Handle document formats with optimal preprocessing for text extraction

Official Dynamic Preprocessing Features (from InternVL3 docs):
1. build_transform(): Official transformation pipeline with proper normalization
2. find_closest_aspect_ratio(): Aspect ratio optimization for multiple tiles
3. dynamic_preprocess(): Official dynamic tiling algorithm (1-12 tiles max)
4. load_image(): Complete preprocessing with proper dtype handling

Key Requirements from Documentation:
- Proper dtype consistency (bfloat16 throughout pipeline)
- ImageNet normalization constants
- BICUBIC interpolation for quality
- Dynamic tiling with thumbnail support
- Memory-safe processing with configurable max_num
"""


# Official ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """
    Official InternVL3 image transformation pipeline

    Args:
        input_size: Target size for image resizing (default 448)

    Returns:
        torchvision.transforms.Compose: Official transformation pipeline
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Official InternVL3 aspect ratio optimization algorithm
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """
    Official InternVL3 dynamic preprocessing algorithm
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    """
    Official InternVL3 image loading with proper dtype handling

    Args:
        image_file: Path to image file (relative to data_dir or absolute)
        input_size: Target size for each tile
        max_num: Maximum number of tiles to generate (1-12 as per docs)

    Returns:
        torch.Tensor: Properly processed image tensor with correct dtype (bfloat16)
    """
    # Handle both relative and absolute paths
    if not image_file.startswith("/"):
        image_file = f"{data_dir}/{image_file}"

    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)

    # CRITICAL: Ensure proper dtype for InternVL3 (must match model's bfloat16)
    return pixel_values.to(torch.bfloat16).cuda()


# Load and process document following official guidelines
document_image = "synthetic_invoice_014.png"  # Configurable document filename
print(f"📄 Loading document from: {data_dir}/{document_image}")

# Load original image for analysis
image_path = f"{data_dir}/{document_image}"
original_image = Image.open(image_path)
print(f"📷 Original document size: {original_image.size}")
print(
    f"📐 Document aspect ratio: {original_image.size[0] / original_image.size[1]:.2f}"
)

# Process with official dynamic preprocessing
print("🖼️  Processing with official InternVL3 dynamic preprocessing...")
pixel_values = load_image(document_image, max_num=12)
print(f"✅ Document processed into {pixel_values.shape[0]} tiles: {pixel_values.shape}")
print(f"🔍 Tensor dtype: {pixel_values.dtype} (should be torch.bfloat16)")
print("📋 Ready for InternVL3 key-value extraction")


"""
Structured Key-Value Extraction Prompt Configuration

Purpose:
- Define comprehensive prompt for extracting structured business document data
- Configure extraction parameters for consistent, standardized output
- Specify exact output format requirements for downstream processing

Extraction Specifications:
- 25 predefined fields covering common business document types
- Supports invoices, receipts, bank statements, and tax documents
- Handles missing fields gracefully with "N/A" placeholders
- Enforces plain text output without markdown formatting
- Ensures deterministic field ordering for automated processing

Field Categories:
1. Document metadata (type, dates)
2. Supplier/business information (name, address, contact)
3. Financial data (amounts, GST, totals)
4. Transaction details (quantities, prices, descriptions)
5. Banking information (account numbers, BSB, balances)

Output Quality Controls:
- Explicit formatting rules to prevent markdown artifacts
- Character limits and validation requirements
- Structured field validation for downstream systems
"""

# Comprehensive key-value extraction prompt optimized for business documents
extraction_prompt = """Extract data from this business document. 
Output ALL fields below with their exact keys. 
Use "N/A" if field is not visible or not present.

REQUIRED OUTPUT FORMAT (output ALL lines exactly as shown):
ABN: [11-digit Australian Business Number or N/A]
ACCOUNT_HOLDER: [value or N/A]
BANK_ACCOUNT_NUMBER: [account number from bank statements only or N/A]
BANK_NAME: [bank name from bank statements only or N/A]
BSB_NUMBER: [6-digit BSB from bank statements only or N/A]
BUSINESS_ADDRESS: [value or N/A]
BUSINESS_PHONE: [value or N/A]
CLOSING_BALANCE: [closing balance amount in dollars or N/A]
DESCRIPTIONS: [list of transaction descriptions or N/A]
DOCUMENT_TYPE: [value or N/A]
DUE_DATE: [value or N/A]
GST: [GST amount in dollars or N/A]
INVOICE_DATE: [value or N/A]
OPENING_BALANCE: [opening balance amount in dollars or N/A]
PAYER_ADDRESS: [value or N/A]
PAYER_EMAIL: [value or N/A]
PAYER_NAME: [value or N/A]
PAYER_PHONE: [value or N/A]
PRICES: [individual prices in dollars or N/A]
QUANTITIES: [list of quantities or N/A]
STATEMENT_PERIOD: [value or N/A]
SUBTOTAL: [subtotal amount in dollars or N/A]
SUPPLIER: [value or N/A]
SUPPLIER_WEBSITE: [value or N/A]
TOTAL: [total amount in dollars or N/A]

CRITICAL: Output in PLAIN TEXT format only. Do NOT use markdown formatting.

CORRECT format: ABN: 12 345 678 901
WRONG format: **ABN:** 12 345 678 901
WRONG format: **ABN: 12 345 678 901**
WRONG format: ABN: **12 345 678 901**

Use exactly: KEY: value (with colon and space)
Never use: **KEY:** or **KEY** or any asterisks
Never use bold, italic, or any markdown formatting

ABSOLUTELY CRITICAL: Output EXACTLY 25 lines using ONLY the keys listed above. 
Do NOT add extra fields like \"Balance\", \"Credit\", \"Debit\", \"Date\", \"Description\".
Do NOT include ANY fields not in the required list above.
Include ALL 25 keys listed above even if value is N/A.
STOP after exactly 25 lines."""

# Format prompt for InternVL3 with proper image token
question = f"<image>\n{extraction_prompt}"

print("📋 Structured key-value extraction prompt configured")
print(f"📄 Prompt length: {len(extraction_prompt)} characters")
print("🔍 Extracting 25 standardized business document fields")
print("⚙️ Configured for deterministic, structured output")


"""
Cell 3.5: Enhanced Batch Processing Infrastructure with Corrected Success Metrics

Purpose:
- Implement batch processing functions with accurate extraction success tracking
- Distinguish between "model returned key" (SUCCESS) vs "model failed to return key" (FAILURE)
- Parse extraction responses into structured dictionaries with success metadata
- Support robust error handling and progress tracking for large datasets

Key Success Metric Fix:
- SUCCESS: Model returns field key with ANY value (including "N/A")
- FAILURE: Model fails to return field key at all or returns malformed output

Batch Processing Components:
1. discover_images(): Find all image files in data_dir with supported formats
2. parse_extraction_response(): Enhanced to track which fields were actually returned
3. process_image_batch(): Main batch processing with correct success tracking
4. create_extraction_dataframe(): Generate pandas DataFrame with proper metadata

Enhanced Features:
- Response completeness tracking (how many keys model returned)
- Content coverage tracking (how many returned keys have non-N/A values)
- Distinction between extraction success and field content availability
- Memory-efficient processing with detailed progress indicators
"""

# Define the 25 extraction fields in alphabetical order for CSV columns
EXTRACTION_FIELDS = [
    "ABN",
    "ACCOUNT_HOLDER",
    "BANK_ACCOUNT_NUMBER",
    "BANK_NAME",
    "BSB_NUMBER",
    "BUSINESS_ADDRESS",
    "BUSINESS_PHONE",
    "CLOSING_BALANCE",
    "DESCRIPTIONS",
    "DOCUMENT_TYPE",
    "DUE_DATE",
    "GST",
    "INVOICE_DATE",
    "OPENING_BALANCE",
    "PAYER_ADDRESS",
    "PAYER_EMAIL",
    "PAYER_NAME",
    "PAYER_PHONE",
    "PRICES",
    "QUANTITIES",
    "STATEMENT_PERIOD",
    "SUBTOTAL",
    "SUPPLIER",
    "SUPPLIER_WEBSITE",
    "TOTAL",
]


def discover_images(directory_path):
    """
    Discover all image files in the specified directory

    Args:
        directory_path (str): Path to directory containing images

    Returns:
        list: List of image file paths found in directory
    """
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    image_files = []

    for extension in image_extensions:
        image_files.extend(str(p) for p in Path(directory_path).glob(extension))

    # Sort for consistent processing order
    image_files.sort()
    return image_files


def parse_extraction_response(response_text):
    """
    Parse extraction response text into structured dictionary with success tracking

    Args:
        response_text (str): Raw text response from InternVL3 extraction

    Returns:
        tuple: (field_dict, extracted_fields_set, success_metadata)
            - field_dict: Dictionary with field names as keys and extracted values
            - extracted_fields_set: Set of fields that model actually returned
            - success_metadata: Dictionary with success statistics
    """
    field_dict = {}
    extracted_fields = set()  # Track which fields model actually returned (SUCCESS)

    # Parse response line by line - don't pre-populate with defaults
    lines = response_text.split("\n")
    for line in lines:
        line = line.strip()
        if ":" in line and not line.startswith("<"):
            try:
                key, value = line.split(":", 1)
                key = key.strip().upper()
                value = value.strip()

                # Only process if it's one of our expected fields
                if key in EXTRACTION_FIELDS:
                    field_dict[key] = value if value else "N/A"
                    extracted_fields.add(key)  # Mark as successfully extracted by model

            except ValueError:
                # Skip malformed lines
                continue

    # Fill missing fields (true failures - model didn't return these keys)
    for field in EXTRACTION_FIELDS:
        if field not in extracted_fields:
            field_dict[field] = "N/A"  # Default value, but marked as failed extraction

    # Calculate success metadata
    successful_extractions = len(extracted_fields)  # Fields model actually returned
    failed_extractions = (
        len(EXTRACTION_FIELDS) - successful_extractions
    )  # Fields model failed to return
    fields_with_content = len(
        [f for f in extracted_fields if field_dict[f] != "N/A"]
    )  # Non-N/A content

    success_metadata = {
        "response_completeness": successful_extractions,  # How many keys model returned
        "response_completeness_rate": (successful_extractions / len(EXTRACTION_FIELDS))
        * 100,
        "content_coverage": fields_with_content,  # How many returned keys have content
        "content_coverage_rate": (fields_with_content / successful_extractions) * 100
        if successful_extractions > 0
        else 0,
        "failed_extractions": failed_extractions,
    }

    return field_dict, extracted_fields, success_metadata


def process_image_batch(image_files, progress_callback=None):
    """
    Process a batch of images through InternVL3 extraction pipeline with enhanced success tracking

    Args:
        image_files (list): List of image file paths to process
        progress_callback (callable, optional): Function to call with progress updates

    Returns:
        tuple: (results, batch_statistics)
            - results: List of dictionaries containing extraction results
            - batch_statistics: Overall batch processing statistics
    """
    results = []
    batch_stats = {
        "total_images": len(image_files),
        "successful_responses": 0,
        "total_fields_returned": 0,
        "total_fields_with_content": 0,
        "processing_errors": 0,
    }

    print(f"🚀 Starting enhanced batch processing of {len(image_files)} images...")

    for i, image_file in enumerate(image_files, 1):
        image_name = Path(image_file).name
        print(f"📷 Processing ({i}/{len(image_files)}): {image_name}")

        try:
            # Load and process image
            pixel_values = load_image(image_file, max_num=12)

            # Execute extraction
            response = model.chat(tokenizer, pixel_values, question, generation_config)

            # Parse response with enhanced success tracking
            extracted_fields, returned_fields, success_meta = parse_extraction_response(
                response
            )

            # Add image name and metadata to results
            result_row = {"image_name": image_name}
            result_row.update(extracted_fields)
            result_row["_response_completeness"] = success_meta[
                "response_completeness"
            ]  # Metadata
            result_row["_content_coverage"] = success_meta[
                "content_coverage"
            ]  # Metadata
            results.append(result_row)

            # Update batch statistics
            batch_stats["successful_responses"] += 1
            batch_stats["total_fields_returned"] += success_meta[
                "response_completeness"
            ]
            batch_stats["total_fields_with_content"] += success_meta["content_coverage"]

            print(
                f"   ✅ Model returned {success_meta['response_completeness']}/25 fields ({success_meta['response_completeness_rate']:.1f}%)"
            )
            print(
                f"   📊 Content in {success_meta['content_coverage']} fields ({success_meta['content_coverage_rate']:.1f}% of returned)"
            )

        except Exception as e:
            print(f"   ❌ Processing error for {image_name}: {str(e)}")

            # Create error result with all N/A values and zero success metrics
            error_result = {"image_name": image_name}
            error_result.update({field: "N/A" for field in EXTRACTION_FIELDS})
            error_result["_response_completeness"] = 0  # Model failed to respond
            error_result["_content_coverage"] = 0  # No content available
            results.append(error_result)

            batch_stats["processing_errors"] += 1

        # Optional progress callback
        if progress_callback:
            progress_callback(i, len(image_files), image_name)

    return results, batch_stats


def create_extraction_dataframe(results):
    """
    Create pandas DataFrame from extraction results with proper column ordering and metadata

    Args:
        results (list): List of dictionaries containing extraction results

    Returns:
        tuple: (df, metadata_df)
            - df: Main DataFrame with image_name + alphabetically ordered field columns
            - metadata_df: DataFrame with success metrics for analysis
    """
    if not results:
        # Create empty DataFrame with proper structure
        columns = ["image_name"] + EXTRACTION_FIELDS
        return pd.DataFrame(columns=columns), pd.DataFrame()

    # Create main DataFrame from results
    results_df = pd.DataFrame(results)

    # Extract metadata columns for separate analysis
    metadata_columns = ["image_name", "_response_completeness", "_content_coverage"]
    metadata_df = (
        results_df[metadata_columns].copy()
        if all(col in results_df.columns for col in metadata_columns)
        else pd.DataFrame()
    )

    # Remove metadata columns from main DataFrame
    main_columns = ["image_name"] + EXTRACTION_FIELDS
    main_df = (
        results_df[main_columns]
        if all(col in results_df.columns for col in main_columns)
        else results_df
    )

    # Ensure proper column ordering: image_name first, then alphabetical fields
    column_order = ["image_name"] + EXTRACTION_FIELDS
    main_df = main_df.reindex(columns=column_order, fill_value="N/A")

    return main_df, metadata_df


print(
    "📋 Enhanced batch processing infrastructure configured with corrected success metrics"
)
print(
    f"🔍 Configured for {len(EXTRACTION_FIELDS)} extraction fields in alphabetical order"
)
print("✅ Success tracking: Model returned keys vs Model failed to return keys")
print("📊 Metrics: Response completeness + Content coverage analysis")
print(
    "⚙️ Functions available: discover_images, parse_extraction_response, process_image_batch, create_extraction_dataframe"
)

"""
Cell 3.6: Ground Truth Loading and Validation Infrastructure

Purpose:
- Load evaluation_ground_truth.csv and create lookup structure for evaluation
- Validate ground truth data completeness and field alignment
- Create image-to-ground-truth mapping for efficient comparison
- Implement sophisticated field-specific accuracy calculation functions

Ground Truth Structure:
- 26 columns: image_file + 25 extraction fields (alphabetically ordered)
- 20 rows: One for each synthetic_invoice_*.png test image
- N/A values for missing/non-applicable fields (consistent with extraction)
- Complex fields use pipe-separated values (QUANTITIES, PRICES, DESCRIPTIONS)

Accuracy Calculation Methods:
- Financial fields (GST, TOTAL, SUBTOTAL): Numeric comparison with 0.01 tolerance
- List fields (QUANTITIES, PRICES, DESCRIPTIONS): Pipe-separated exact matching
- Date fields (INVOICE_DATE, DUE_DATE): Flexible format comparison
- String fields: Fuzzy matching (exact=1.0, partial=0.8, none=0.0)
- Missing value handling: Proper N/A vs empty comparison
"""


def load_ground_truth(csv_path):
    """
    Load ground truth CSV and create image-to-ground-truth mapping

    Args:
        csv_path (str): Path to evaluation_ground_truth.csv

    Returns:
        dict: Mapping of image_file to ground truth field dictionary
    """
    try:
        # Load ground truth CSV
        gt_df = pd.read_csv(csv_path)
        print(
            f"📊 Loaded ground truth: {len(gt_df)} rows × {len(gt_df.columns)} columns"
        )

        # Validate structure
        expected_columns = ["image_file"] + EXTRACTION_FIELDS
        actual_columns = list(gt_df.columns)

        if len(actual_columns) != len(expected_columns):
            print(
                f"⚠️ Column count mismatch: expected {len(expected_columns)}, got {len(actual_columns)}"
            )

        # Check field alignment
        missing_fields = set(expected_columns) - set(actual_columns)
        extra_fields = set(actual_columns) - set(expected_columns)

        if missing_fields:
            print(f"⚠️ Missing fields in ground truth: {missing_fields}")
        if extra_fields:
            print(f"⚠️ Extra fields in ground truth: {extra_fields}")

        # Create image-to-ground-truth mapping
        ground_truth_map = {}
        for _, row in gt_df.iterrows():
            image_file = row["image_file"]
            gt_data = {
                field: str(row[field]) if pd.notna(row[field]) else "N/A"
                for field in EXTRACTION_FIELDS
                if field in row.index
            }
            ground_truth_map[image_file] = gt_data

        print(f"✅ Created ground truth mapping for {len(ground_truth_map)} images")

        # Display sample for verification
        sample_image = list(ground_truth_map.keys())[0]
        sample_gt = ground_truth_map[sample_image]
        print(f"\n📋 Sample ground truth for {sample_image}:")
        for field, value in list(sample_gt.items())[:5]:  # Show first 5 fields
            print(f"   {field}: {value}")
        print(f"   ... and {len(sample_gt) - 5} more fields")

        return ground_truth_map

    except FileNotFoundError:
        print(f"❌ Ground truth file not found: {csv_path}")
        return {}
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return {}


def calculate_field_accuracy(extracted_value, ground_truth_value, field_name):
    """
    Calculate accuracy for a specific field using sophisticated comparison logic

    Args:
        extracted_value (str): Value extracted by the model
        ground_truth_value (str): Correct value from ground truth
        field_name (str): Name of the field being compared

    Returns:
        float: Accuracy score between 0.0 and 1.0
    """
    # Handle missing values - FIXED to include 'nan' and other variants
    na_variants = ["N/A", "NA", "", "NAN", "NULL", "NONE", "NIL"]
    # Handle missing values
    if not ground_truth_value or str(ground_truth_value).upper() in na_variants:
        return (
            1.0
            if (not extracted_value or str(extracted_value).upper() in na_variants)
            else 0.0
        )

    if not extracted_value or extracted_value.upper() in na_variants:
        return 0.0  # Ground truth exists but nothing extracted

    # Normalize for comparison
    extracted_clean = str(extracted_value).strip()
    gt_clean = str(ground_truth_value).strip()

    # Exact match (case-insensitive)
    if extracted_clean.lower() == gt_clean.lower():
        return 1.0

    # Field-specific comparison logic
    if field_name in ["GST", "TOTAL", "SUBTOTAL", "OPENING_BALANCE", "CLOSING_BALANCE"]:
        # Numeric comparison with tolerance for financial fields
        try:
            # Extract numeric values (remove currency symbols, commas, etc.)
            ext_num = float(re.sub(r"[^\d.-]", "", extracted_clean.replace(",", "")))
            gt_num = float(re.sub(r"[^\d.-]", "", gt_clean.replace(",", "")))

            # Allow small tolerance for floating point precision
            tolerance = 0.01
            return 1.0 if abs(ext_num - gt_num) < tolerance else 0.0

        except (ValueError, TypeError):
            # Fallback to string comparison if numeric parsing fails
            return 1.0 if extracted_clean.lower() == gt_clean.lower() else 0.0

    elif field_name in ["QUANTITIES", "PRICES", "DESCRIPTIONS"]:
        # List comparison for pipe-separated values
        try:
            ext_items = [item.strip() for item in extracted_clean.split("|")]
            gt_items = [item.strip() for item in gt_clean.split("|")]

            # Must have same number of items
            if len(ext_items) != len(gt_items):
                return 0.0

            # Calculate item-wise matches
            matches = sum(
                1
                for e, g in zip(ext_items, gt_items, strict=False)
                if e.lower().strip() == g.lower().strip()
            )

            return matches / len(gt_items) if gt_items else 0.0

        except Exception:
            # Fallback to string comparison
            return 1.0 if extracted_clean.lower() == gt_clean.lower() else 0.0

    elif field_name in ["INVOICE_DATE", "DUE_DATE"]:
        # Date comparison with flexible format handling
        try:
            # Extract date components (numbers, slashes, dashes)
            ext_date = re.sub(r"[^\d/\-]", "", extracted_clean)
            gt_date = re.sub(r"[^\d/\-]", "", gt_clean)

            return 1.0 if ext_date == gt_date else 0.0

        except Exception:
            return 1.0 if extracted_clean.lower() == gt_clean.lower() else 0.0

    else:
        # String comparison with fuzzy matching for other fields
        if extracted_clean.lower() == gt_clean.lower():
            return 1.0
        elif (
            extracted_clean.lower() in gt_clean.lower()
            or gt_clean.lower() in extracted_clean.lower()
        ):
            return 0.8  # Partial match
        else:
            return 0.0


def evaluate_extraction_results(extraction_results, ground_truth_map):
    """
    Evaluate extraction results against ground truth with comprehensive metrics

    Args:
        extraction_results (list): List of extraction result dictionaries
        ground_truth_map (dict): Ground truth mapping by image filename

    Returns:
        dict: Comprehensive evaluation metrics and analysis
    """
    if not extraction_results or not ground_truth_map:
        return {"error": "Missing extraction results or ground truth data"}

    evaluation_data = []
    field_accuracies = {field: [] for field in EXTRACTION_FIELDS}
    overall_accuracies = []

    print(
        f"🎯 Evaluating {len(extraction_results)} extraction results against ground truth..."
    )

    for i, result in enumerate(extraction_results, 1):
        image_name = result["image_name"]
        print(f"📊 Evaluating ({i}/{len(extraction_results)}): {image_name}")

        # Get ground truth for this image
        gt_data = ground_truth_map.get(image_name, {})

        if not gt_data:
            print(f"   ⚠️ No ground truth found for {image_name}")
            continue

        # Calculate field-wise accuracies
        image_evaluation = {"image_name": image_name}
        image_field_accuracies = {}

        for field in EXTRACTION_FIELDS:
            extracted_value = result.get(field, "N/A")
            gt_value = gt_data.get(field, "N/A")

            accuracy = calculate_field_accuracy(extracted_value, gt_value, field)
            image_field_accuracies[field] = accuracy
            field_accuracies[field].append(accuracy)

            # Store both extracted value, ground truth, and accuracy
            image_evaluation[f"{field}_extracted"] = extracted_value
            image_evaluation[f"{field}_ground_truth"] = gt_value
            image_evaluation[f"{field}_accuracy"] = accuracy

        # Calculate overall accuracy for this image
        image_accuracy = sum(image_field_accuracies.values()) / len(
            image_field_accuracies
        )
        image_evaluation["overall_accuracy"] = image_accuracy
        overall_accuracies.append(image_accuracy)

        evaluation_data.append(image_evaluation)

        # Show progress
        fields_correct = sum(
            1 for acc in image_field_accuracies.values() if acc >= 0.99
        )
        print(
            f"   ✅ {fields_correct}/25 fields correct ({image_accuracy:.1%} accuracy)"
        )

    # Calculate aggregate metrics
    evaluation_summary = {
        "total_images": len(evaluation_data),
        "overall_accuracy": sum(overall_accuracies) / len(overall_accuracies)
        if overall_accuracies
        else 0.0,
        "perfect_documents": sum(1 for acc in overall_accuracies if acc >= 0.99),
        "field_accuracies": {
            field: sum(accs) / len(accs) if accs else 0.0
            for field, accs in field_accuracies.items()
        },
        "evaluation_data": evaluation_data,
    }

    return evaluation_summary


# Load ground truth data
print("📊 Loading ground truth data for evaluation...")
ground_truth_data = load_ground_truth(ground_truth_path)

if ground_truth_data:
    print(f"✅ Ground truth loaded successfully for {len(ground_truth_data)} images")
    print("🎯 Evaluation infrastructure ready")
else:
    print("❌ Failed to load ground truth data - evaluation will be limited")


"""
Enhanced Batch Key-Value Extraction with Corrected Success Metrics

Purpose:
- Execute batch processing across all images in data_dir with accurate success tracking
- Generate comprehensive CSV output with corrected extraction metrics
- Provide detailed statistics distinguishing response completeness from content coverage

Enhanced Execution Pipeline:
1. Discover all image files in the configured data directory
2. Process each image through InternVL3 extraction pipeline with success tracking
3. Parse and structure extraction results with metadata about model performance
4. Create pandas DataFrame with proper column ordering and success analysis
5. Export results to CSV with comprehensive metadata and corrected statistics

Corrected Success Metrics:
- Response Completeness: How many field keys the model actually returned
- Content Coverage: How many returned keys have non-N/A values
- True Success Rate: Based on model returning keys, not field content

Output Format:
- CSV Structure: image_name + 25 alphabetically ordered field columns
- Metadata tracking: Response completeness and content coverage per image
- File Location: {output_dir}/internvl3_batch_extraction.csv
- Enhanced statistics: Accurate model performance assessment
"""

# Generation configuration optimized for structured output
generation_config = dict(
    max_new_tokens=1000,  # Adequate tokens for 25 structured fields
    do_sample=False,  # Deterministic for consistent field extraction
    pad_token_id=tokenizer.eos_token_id,  # Prevent pad_token_id warnings
)

print("🔍 Discovering images in data directory...")
image_files = discover_images(data_dir)

if not image_files:
    print(f"❌ No image files found in {data_dir}")
    print("💡 Supported formats: PNG, JPG, JPEG (case insensitive)")
else:
    print(f"✅ Found {len(image_files)} image files to process")

    # Show sample of files that will be processed
    print("\n📋 Sample of files to be processed:")
    for i, file_path in enumerate(image_files[:5]):
        print(f"   {i + 1}. {Path(file_path).name}")
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more files")

    print("\n🚀 Starting enhanced batch key-value extraction with corrected metrics...")
    start_time = datetime.now()

    try:
        # Process all images through enhanced extraction pipeline
        extraction_results, batch_statistics = process_image_batch(image_files)

        # Create structured DataFrame with proper column ordering and metadata
        print("\n📊 Creating structured DataFrame with success metrics...")
        df, metadata_df = create_extraction_dataframe(extraction_results)

        print(
            f"✅ Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        print(
            f"📋 Column structure: image_name + {len(EXTRACTION_FIELDS)} alphabetically ordered fields"
        )

        # Generate CSV output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = "internvl3_batch_extraction.csv"
        csv_filename_timestamped = f"internvl3_batch_extraction_{timestamp}.csv"

        csv_path = Path(output_dir) / csv_filename
        csv_path_timestamped = Path(output_dir) / csv_filename_timestamped

        # Ensure output directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Save main CSV file (without metadata columns)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"💾 Main CSV saved: {csv_path}")

        # Save timestamped backup
        df.to_csv(csv_path_timestamped, index=False, encoding="utf-8")
        print(f"📄 Backup CSV saved: {csv_path_timestamped}")

        # Save metadata for analysis (optional detailed file)
        if not metadata_df.empty:
            metadata_path = (
                Path(output_dir) / f"internvl3_extraction_metadata_{timestamp}.csv"
            )
            metadata_df.to_csv(metadata_path, index=False, encoding="utf-8")
            print(f"📈 Metadata saved: {metadata_path}")

        # Enhanced processing statistics with corrected metrics
        end_time = datetime.now()
        processing_duration = end_time - start_time

        print("\n📈 Enhanced Batch Processing Summary:")
        print(f"   • Total images processed: {batch_statistics['total_images']}")
        print(
            f"   • Successful model responses: {batch_statistics['successful_responses']}"
        )
        print(f"   • Processing errors: {batch_statistics['processing_errors']}")
        print(f"   • Processing duration: {processing_duration}")
        print(
            f"   • Average time per image: {processing_duration.total_seconds() / len(extraction_results):.2f} seconds"
        )

        # Corrected success rate analysis
        total_possible_fields = len(extraction_results) * len(EXTRACTION_FIELDS)
        overall_response_rate = (
            batch_statistics["total_fields_returned"] / total_possible_fields
        ) * 100
        overall_content_rate = (
            (
                batch_statistics["total_fields_with_content"]
                / batch_statistics["total_fields_returned"]
            )
            * 100
            if batch_statistics["total_fields_returned"] > 0
            else 0
        )

        print("\n🎯 Corrected Extraction Performance Metrics:")
        print(f"   • Total possible field extractions: {total_possible_fields:,}")
        print(
            f"   • Total fields returned by model: {batch_statistics['total_fields_returned']:,}"
        )
        print(
            f"   • Model response completeness: {overall_response_rate:.1f}% (fields returned)"
        )
        print(
            f"   • Content coverage of returned fields: {overall_content_rate:.1f}% (non-N/A values)"
        )

        # Per-image performance analysis
        if not metadata_df.empty:
            avg_response_completeness = metadata_df["_response_completeness"].mean()
            avg_content_coverage = metadata_df["_content_coverage"].mean()
            max_response = metadata_df["_response_completeness"].max()
            min_response = metadata_df["_response_completeness"].min()

            print("\n📷 Per-Image Analysis (Corrected Metrics):")
            print(
                f"   • Average fields returned per image: {avg_response_completeness:.1f}/{len(EXTRACTION_FIELDS)}"
            )
            print(f"   • Average content fields per image: {avg_content_coverage:.1f}")
            print(
                f"   • Best model response: {max_response}/{len(EXTRACTION_FIELDS)} fields returned"
            )
            print(
                f"   • Worst model response: {min_response}/{len(EXTRACTION_FIELDS)} fields returned"
            )

            # Find best and worst performing images
            best_idx = metadata_df["_response_completeness"].idxmax()
            worst_idx = metadata_df["_response_completeness"].idxmin()

            best_image = metadata_df.loc[best_idx, "image_name"]
            worst_image = metadata_df.loc[worst_idx, "image_name"]

            print(f"   • Best performing image: {best_image} ({max_response} fields)")
            print(f"   • Worst performing image: {worst_image} ({min_response} fields)")

        # File size information
        csv_size = csv_path.stat().st_size
        print("\n📊 Output File Statistics:")
        print(f"   • Main CSV file size: {csv_size:,} bytes")
        print(
            f"   • Images with perfect model response (25/25): {len([r for r in extraction_results if r.get('_response_completeness', 0) == 25])}"
        )
        print(
            f"   • Images with partial model response (1-24): {len([r for r in extraction_results if 0 < r.get('_response_completeness', 0) < 25])}"
        )
        print(
            f"   • Images with no model response (0): {len([r for r in extraction_results if r.get('_response_completeness', 0) == 0])}"
        )

        print("\n🚀 Enhanced batch processing completed successfully!")
        print(f"📁 Results available at: {csv_path}")
        print(
            "✅ Success metrics now correctly distinguish model response from field content"
        )

    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

        import traceback

        print("\n🔧 Full error traceback:")
        traceback.print_exc()


"""
Enhanced Data Analysis with Corrected Success Metrics

Purpose:
- Perform comprehensive analysis of CSV extraction results with accurate success tracking
- Distinguish between Response Completeness (model returned keys) and Content Coverage (keys have values)
- Validate data quality and provide insights for workflow optimization
- Generate corrected field-level statistics and performance reports

Corrected Analysis Components:
1. CSV file validation and structural analysis
2. Response completeness analysis (model performance)
3. Content coverage analysis (field availability in documents)
4. Enhanced quality metrics with proper success/failure classification

Enhanced Quality Metrics:
- Response Completeness: How many field keys the model actually returned (TRUE SUCCESS)
- Content Coverage: How many returned keys have non-N/A values (CONTENT AVAILABILITY)
- Field-level success rates: Which fields model consistently returns vs fails to return
- Processing efficiency and error rate analysis with accurate classifications

Integration Support:
- CSV structure validation for downstream systems
- Corrected success rate reporting for model evaluation
- Enhanced workflow optimization recommendations
- Accurate extraction performance assessment
"""

try:
    # Verify CSV file exists and load for analysis
    csv_path = Path(output_dir) / "internvl3_batch_extraction.csv"

    if not csv_path.exists():
        print("❌ CSV file not found. Please run the batch processing cell first.")
    else:
        print("📊 Loading CSV file for enhanced analysis with corrected metrics...")
        df_analysis = pd.read_csv(csv_path)

        print(
            f"✅ CSV loaded successfully: {len(df_analysis)} rows × {len(df_analysis.columns)} columns"
        )

        # Basic CSV structure validation
        print("\n🔍 CSV Structure Analysis:")
        print(f"   • File size: {csv_path.stat().st_size:,} bytes")
        print(f"   • Number of images processed: {len(df_analysis)}")
        print(
            f"   • Number of extraction fields: {len(df_analysis.columns) - 1}"
        )  # -1 for image_name column

        # Column validation
        expected_columns = ["image_name"] + EXTRACTION_FIELDS
        actual_columns = list(df_analysis.columns)

        if actual_columns == expected_columns:
            print("   ✅ Column structure matches expected format")
        else:
            print("   ⚠️ Column structure differs from expected format")
            print(f"      Expected: {len(expected_columns)} columns")
            print(f"      Actual: {len(actual_columns)} columns")

        # Load metadata if available for enhanced analysis
        metadata_files = list(
            Path(output_dir).glob("internvl3_extraction_metadata_*.csv")
        )
        metadata_df = None

        if metadata_files:
            latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
            metadata_df = pd.read_csv(latest_metadata)
            print(f"   📈 Metadata loaded from: {latest_metadata.name}")

        # Enhanced field-level analysis with corrected metrics
        print("\n📈 Enhanced Field-Level Analysis (Corrected Success Metrics):")

        if metadata_df is not None and len(metadata_df) == len(df_analysis):
            # Use metadata for accurate success tracking
            print("   🎯 Using enhanced success tracking from metadata")

            # Calculate average response completeness and content coverage
            avg_response_completeness = metadata_df["_response_completeness"].mean()
            avg_content_coverage = metadata_df["_content_coverage"].mean()

            print("\n   📊 Overall Model Performance:")
            print(
                f"      • Average fields returned per image: {avg_response_completeness:.1f}/{len(EXTRACTION_FIELDS)}"
            )
            print(
                f"      • Model response completeness: {(avg_response_completeness / len(EXTRACTION_FIELDS)) * 100:.1f}%"
            )
            print(
                f"      • Average content fields per image: {avg_content_coverage:.1f}"
            )
            print(
                f"      • Content coverage rate: {(avg_content_coverage / avg_response_completeness) * 100 if avg_response_completeness > 0 else 0:.1f}%"
            )

            # Distribution analysis
            perfect_responses = len(
                metadata_df[metadata_df["_response_completeness"] == 25]
            )
            partial_responses = len(
                metadata_df[
                    (metadata_df["_response_completeness"] > 0)
                    & (metadata_df["_response_completeness"] < 25)
                ]
            )
            failed_responses = len(
                metadata_df[metadata_df["_response_completeness"] == 0]
            )

            print("\n   📊 Response Completeness Distribution:")
            print(
                f"      • Perfect responses (25/25 fields): {perfect_responses} ({perfect_responses / len(metadata_df) * 100:.1f}%)"
            )
            print(
                f"      • Partial responses (1-24 fields): {partial_responses} ({partial_responses / len(metadata_df) * 100:.1f}%)"
            )
            print(
                f"      • Failed responses (0 fields): {failed_responses} ({failed_responses / len(metadata_df) * 100:.1f}%)"
            )

        # Field-by-field analysis (traditional method as fallback)
        print("\n   📊 Field-by-Field Content Analysis:")
        field_stats = {}

        for field in EXTRACTION_FIELDS:
            if field in df_analysis.columns:
                non_na_count = len(df_analysis[df_analysis[field] != "N/A"])
                content_rate = (non_na_count / len(df_analysis)) * 100
                field_stats[field] = {
                    "content_count": non_na_count,
                    "content_rate": content_rate,
                    "total_images": len(df_analysis),
                }

        # Sort fields by content availability for insights
        sorted_fields = sorted(
            field_stats.items(), key=lambda x: x[1]["content_rate"], reverse=True
        )

        print("      📈 Top 10 Fields with Most Content Available:")
        for i, (field, stats) in enumerate(sorted_fields[:10], 1):
            print(
                f"         {i:2d}. {field:<20} {stats['content_count']:3d}/{stats['total_images']:3d} ({stats['content_rate']:5.1f}% content)"
            )

        print("\n      📉 Bottom 5 Fields with Least Content Available:")
        for i, (field, stats) in enumerate(sorted_fields[-5:], len(sorted_fields) - 4):
            print(
                f"         {i:2d}. {field:<20} {stats['content_count']:3d}/{stats['total_images']:3d} ({stats['content_rate']:5.1f}% content)"
            )

        # Overall extraction performance with corrected interpretation
        total_possible_content = len(df_analysis) * len(EXTRACTION_FIELDS)
        total_available_content = sum(
            stats["content_count"] for stats in field_stats.values()
        )
        overall_content_availability = (
            total_available_content / total_possible_content
        ) * 100

        print("\n🎯 Corrected Performance Analysis:")
        print(f"   • Total possible field instances: {total_possible_content:,}")
        print(f"   • Total content available: {total_available_content:,}")
        print(f"   • Overall content availability: {overall_content_availability:.1f}%")
        print("   • Note: This measures document content, not model extraction success")

        # Enhanced image-level analysis
        image_content_performance = []
        for _, row in df_analysis.iterrows():
            content_fields = sum(
                1 for field in EXTRACTION_FIELDS if row[field] != "N/A"
            )
            image_content_performance.append(content_fields)

        avg_content_per_image = sum(image_content_performance) / len(
            image_content_performance
        )
        max_content = max(image_content_performance)
        min_content = min(image_content_performance)

        print("\n📷 Per-Image Content Analysis:")
        print(
            f"   • Average content fields per image: {avg_content_per_image:.1f}/{len(EXTRACTION_FIELDS)}"
        )
        print(
            f"   • Richest document: {max_content}/{len(EXTRACTION_FIELDS)} fields with content"
        )
        print(
            f"   • Sparsest document: {min_content}/{len(EXTRACTION_FIELDS)} fields with content"
        )

        # Identify richest and sparsest documents
        df_with_content_performance = df_analysis.copy()
        df_with_content_performance["content_fields"] = image_content_performance

        richest_image = df_with_content_performance.loc[
            df_with_content_performance["content_fields"].idxmax()
        ]
        sparsest_image = df_with_content_performance.loc[
            df_with_content_performance["content_fields"].idxmin()
        ]

        print(
            f"   • Richest document: {richest_image['image_name']} ({richest_image['content_fields']} fields)"
        )
        print(
            f"   • Sparsest document: {sparsest_image['image_name']} ({sparsest_image['content_fields']} fields)"
        )

        # Enhanced data quality validation
        print("\n🔧 Enhanced Data Quality Validation:")

        # Check for completely empty documents (all N/A)
        empty_documents = df_analysis[
            df_analysis[EXTRACTION_FIELDS].eq("N/A").all(axis=1)
        ]
        print(f"   • Documents with no content: {len(empty_documents)}")

        if len(empty_documents) > 0:
            print("     📄 Documents with no extractable content:")
            for _idx, row in empty_documents.head(3).iterrows():
                print(f"       - {row['image_name']}")
            if len(empty_documents) > 3:
                print(f"       ... and {len(empty_documents) - 3} more")

        # Integration readiness assessment with corrected understanding
        print("\n🚀 Enhanced Integration Readiness Assessment:")

        if metadata_df is not None:
            # Use model performance metrics for readiness assessment
            model_success_rate = (
                avg_response_completeness / len(EXTRACTION_FIELDS)
            ) * 100

            if model_success_rate >= 95:
                print("   ✅ EXCELLENT - Model consistently returns structured fields")
            elif model_success_rate >= 80:
                print("   ✅ GOOD - Model reliably processes extraction requests")
            elif model_success_rate >= 60:
                print(
                    "   ⚠️ FAIR - Model partially processes requests, some optimization needed"
                )
            else:
                print("   ❌ POOR - Model struggles with structured extraction format")

            print(
                f"   📊 Model Performance: {model_success_rate:.1f}% field return rate"
            )
        else:
            # Fallback to content-based assessment
            if overall_content_availability >= 50:
                print("   ✅ GOOD - Documents contain substantial extractable content")
            elif overall_content_availability >= 30:
                print("   ⚠️ FAIR - Documents have moderate content availability")
            else:
                print("   ❌ SPARSE - Documents contain limited extractable content")

        print("\n📁 Enhanced CSV File Details:")
        print(f"   • Main file: {csv_path}")
        print("   • File format: UTF-8 encoded CSV")
        print("   • Column separator: comma (,)")
        print("   • Missing value representation: N/A")
        print(
            "   • Success metrics: Corrected to distinguish model performance from content availability"
        )
        print("   • Ready for: Database import, spreadsheet analysis, API integration")

        print("\n✅ Enhanced data analysis completed with corrected success metrics!")
        print(
            "📈 Key insight: Success now properly measured by model field return rate, not content availability"
        )

except FileNotFoundError:
    print("❌ CSV file not found for analysis")
    print("💡 Please run the batch processing cell first to generate the CSV")

except Exception as e:
    print(f"❌ Error during enhanced data analysis: {e}")
    print(f"🔍 Error type: {type(e).__name__}")

    import traceback

    print("\n🔧 Full error traceback:")
    traceback.print_exc()


"""
Advanced Data Analysis and CSV Validation

Purpose:
- Perform comprehensive analysis of the generated CSV extraction results
- Validate data quality and provide insights for workflow optimization
- Generate detailed field-level statistics and coverage reports

Analysis Components:
1. CSV file validation and structural analysis
2. Field-by-field coverage and content analysis
3. Data quality metrics and extraction performance
4. Export validation and integration readiness assessment

Quality Metrics:
- Field completeness rates across all processed images
- Most/least successfully extracted field types
- Data consistency and format validation
- Processing efficiency and error rate analysis

Integration Support:
- CSV structure validation for downstream systems
- Data type consistency verification
- Missing value pattern analysis
- Export format compliance checking
"""

try:
    # Verify CSV file exists and load for analysis
    csv_path = Path(output_dir) / "internvl3_batch_extraction.csv"

    if not csv_path.exists():
        print("❌ CSV file not found. Please run the batch processing cell first.")
    else:
        print("📊 Loading CSV file for comprehensive analysis...")
        df_analysis = pd.read_csv(csv_path)

        print(
            f"✅ CSV loaded successfully: {len(df_analysis)} rows × {len(df_analysis.columns)} columns"
        )

        # Basic CSV structure validation
        print("\n🔍 CSV Structure Analysis:")
        print(f"   • File size: {csv_path.stat().st_size:,} bytes")
        print(f"   • Number of images processed: {len(df_analysis)}")
        print(
            f"   • Number of extraction fields: {len(df_analysis.columns) - 1}"
        )  # -1 for image_name column

        # Column validation
        expected_columns = ["image_name"] + EXTRACTION_FIELDS
        actual_columns = list(df_analysis.columns)

        if actual_columns == expected_columns:
            print("   ✅ Column structure matches expected format")
        else:
            print("   ⚠️ Column structure differs from expected format")
            print(f"      Expected: {len(expected_columns)} columns")
            print(f"      Actual: {len(actual_columns)} columns")

        # Field-level analysis
        print("\n📈 Field-Level Coverage Analysis:")
        field_stats = {}

        for field in EXTRACTION_FIELDS:
            if field in df_analysis.columns:
                non_na_count = len(df_analysis[df_analysis[field] != "N/A"])
                coverage_rate = (non_na_count / len(df_analysis)) * 100
                field_stats[field] = {
                    "coverage_count": non_na_count,
                    "coverage_rate": coverage_rate,
                    "total_images": len(df_analysis),
                }

        # Sort fields by coverage rate for better insights
        sorted_fields = sorted(
            field_stats.items(), key=lambda x: x[1]["coverage_rate"], reverse=True
        )

        print("   📊 Top 10 Most Successfully Extracted Fields:")
        for i, (field, stats) in enumerate(sorted_fields[:10], 1):
            print(
                f"      {i:2d}. {field:<20} {stats['coverage_count']:3d}/{stats['total_images']:3d} ({stats['coverage_rate']:5.1f}%)"
            )

        print("\n   📊 Bottom 5 Least Successfully Extracted Fields:")
        for i, (field, stats) in enumerate(sorted_fields[-5:], len(sorted_fields) - 4):
            print(
                f"      {i:2d}. {field:<20} {stats['coverage_count']:3d}/{stats['total_images']:3d} ({stats['coverage_rate']:5.1f}%)"
            )

        # Overall extraction performance
        total_possible_fields = len(df_analysis) * len(EXTRACTION_FIELDS)
        total_extracted_fields = sum(
            stats["coverage_count"] for stats in field_stats.values()
        )
        overall_extraction_rate = (total_extracted_fields / total_possible_fields) * 100

        print("\n🎯 Overall Extraction Performance:")
        print(f"   • Total possible field extractions: {total_possible_fields:,}")
        print(f"   • Total successful field extractions: {total_extracted_fields:,}")
        print(f"   • Overall extraction success rate: {overall_extraction_rate:.1f}%")

        # Image-level analysis
        image_performance = []
        for _, row in df_analysis.iterrows():
            extracted_fields = sum(
                1 for field in EXTRACTION_FIELDS if row[field] != "N/A"
            )
            image_performance.append(extracted_fields)

        avg_fields_per_image = sum(image_performance) / len(image_performance)
        max_fields = max(image_performance)
        min_fields = min(image_performance)

        print("\n📷 Per-Image Extraction Analysis:")
        print(
            f"   • Average fields extracted per image: {avg_fields_per_image:.1f}/{len(EXTRACTION_FIELDS)}"
        )
        print(
            f"   • Best performing image: {max_fields}/{len(EXTRACTION_FIELDS)} fields"
        )
        print(
            f"   • Worst performing image: {min_fields}/{len(EXTRACTION_FIELDS)} fields"
        )

        # Identify best and worst performing images
        df_with_performance = df_analysis.copy()
        df_with_performance["extracted_fields"] = image_performance

        best_image = df_with_performance.loc[
            df_with_performance["extracted_fields"].idxmax()
        ]
        worst_image = df_with_performance.loc[
            df_with_performance["extracted_fields"].idxmin()
        ]

        print(
            f"   • Best performing image: {best_image['image_name']} ({best_image['extracted_fields']} fields)"
        )
        print(
            f"   • Worst performing image: {worst_image['image_name']} ({worst_image['extracted_fields']} fields)"
        )

        # Data consistency checks
        print("\n🔧 Data Quality Validation:")

        # Check for completely empty rows (all N/A except image_name)
        empty_rows = df_analysis[df_analysis[EXTRACTION_FIELDS].eq("N/A").all(axis=1)]
        print(f"   • Images with no extracted fields: {len(empty_rows)}")

        if len(empty_rows) > 0:
            print("     Failed images:")
            for _idx, row in empty_rows.head(3).iterrows():
                print(f"       - {row['image_name']}")
            if len(empty_rows) > 3:
                print(f"       ... and {len(empty_rows) - 3} more")

        # Integration readiness assessment
        print("\n🚀 Integration Readiness Assessment:")
        if overall_extraction_rate >= 80:
            print(
                "   ✅ EXCELLENT - High extraction success rate, ready for production"
            )
        elif overall_extraction_rate >= 60:
            print(
                "   ✅ GOOD - Acceptable extraction rate, suitable for most applications"
            )
        elif overall_extraction_rate >= 40:
            print("   ⚠️ FAIR - Moderate extraction rate, consider process optimization")
        else:
            print(
                "   ❌ POOR - Low extraction rate, requires investigation and improvement"
            )

        print("\n📁 CSV File Details:")
        print(f"   • Main file: {csv_path}")
        print("   • File format: UTF-8 encoded CSV")
        print("   • Column separator: comma (,)")
        print("   • Missing value representation: N/A")
        print("   • Ready for: Database import, spreadsheet analysis, API integration")

        print("\n✅ Data analysis completed successfully!")

except FileNotFoundError:
    print("❌ CSV file not found for analysis")
    print("💡 Please run the batch processing cell first to generate the CSV")

except Exception as e:
    print(f"❌ Error during data analysis: {e}")
    print(f"🔍 Error type: {type(e).__name__}")

    import traceback

    print("\n🔧 Full error traceback:")
    traceback.print_exc()


"""
Results Saving and Analysis Pipeline

Purpose:
- Save extracted key-value pairs to persistent storage for further processing
- Perform quality analysis and validation of extraction results
- Generate extraction reports and statistics for workflow integration

File Operations:
- Creates output directory using global output_dir configuration
- Uses UTF-8 encoding for proper international character handling
- Saves with descriptive filename including timestamp capability
- Implements atomic file operations to prevent data corruption

Quality Analysis Features:
- Field completeness assessment (target: 25 fields)
- Content validation for required field formats
- Data quality indicators for downstream processing
- Extraction confidence metrics and reporting

Error Handling:
- NameError: Handles case where response variable isn't defined
- FileSystem errors: Permission issues, disk space, path problems
- Encoding errors: Character set and formatting issues
- Provides actionable troubleshooting guidance for each error type

Integration Features:
- Structured output suitable for database import
- JSON-compatible field parsing for API integration
- Batch processing support for multiple document workflows
"""

# Configure output path using global output_dir variable
output_filename = "internvl3_keyvalue_extraction.txt"
output_path = Path(output_dir) / output_filename

print(f"💾 Saving extraction results to: {output_path}")

try:
    # Ensure output directory exists with proper permissions
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write extraction results with UTF-8 encoding for international support
    with output_path.open("w", encoding="utf-8") as text_file:
        text_file.write(response)

    print("✅ Key-value extraction results saved successfully!")
    print(f"📄 File location: {output_path}")
    print(f"📊 File size: {output_path.stat().st_size} bytes")

    # Advanced extraction analysis and reporting
    lines = response.split("\n")
    field_lines = [
        line for line in lines if ":" in line and not line.strip().startswith("<")
    ]

    print("\n📈 Detailed Extraction Analysis:")
    print(f"   • Document processed: {document_image}")
    print(f"   • Total response lines: {len(lines)}")
    print(f"   • Structured field lines: {len(field_lines)}")
    print(f"   • Field extraction rate: {len(field_lines) / 25 * 100:.1f}%")

    # Field content analysis
    non_na_fields = [
        line
        for line in field_lines
        if line.split(":")[1].strip().upper() not in ["N/A", "NA"]
    ]
    print(f"   • Fields with content: {len(non_na_fields)}")
    print(f"   • Content coverage: {len(non_na_fields) / 25 * 100:.1f}%")

    # File validation
    file_size = output_path.stat().st_size
    if file_size > 100:
        print("✅ Output file validation: PASSED (sufficient content)")
    else:
        print("⚠️ Output file validation: WARNING (minimal content detected)")

    print("\n🔗 Integration ready: Results saved in structured format")
    print(f"📁 Output directory: {output_dir}")

except NameError:
    print("❌ Error: Extraction response not available")
    print("💡 Solution: Execute Cell 4 first to generate extraction results")
    print("🔄 Then re-run this cell to save the results")

except PermissionError:
    print(f"❌ Permission Error: Cannot write to {output_path}")
    print("💡 Solutions:")
    print("   • Check directory write permissions")
    print("   • Verify output_dir path is accessible")
    print("   • Try running with appropriate user permissions")

except OSError as e:
    print(f"❌ File System Error: {e}")
    print("💡 Solutions:")
    print("   • Check available disk space")
    print("   • Verify path validity and accessibility")
    print("   • Ensure parent directories exist")

except Exception as e:
    print(f"❌ Unexpected error during file operations: {e}")
    print(f"🔍 Error type: {type(e).__name__}")
    print("💡 Check system resources and file path configuration")
    print(f"🗂️ Configured output directory: {output_dir}")

    import traceback

    print("\n🔧 Full error details:")
    traceback.print_exc()


"""
Ground Truth Evaluation and Performance Analysis

Purpose:
- Load extraction results and perform comprehensive evaluation against ground truth
- Calculate field-level and document-level accuracy using sophisticated comparison methods
- Generate detailed evaluation metrics distinguishing extraction quality from model performance
- Export evaluation results and performance analytics for rigorous model assessment

Evaluation Components:
1. Load extraction results CSV and ground truth data
2. Perform field-by-field accuracy comparison with sophisticated matching rules
3. Calculate comprehensive metrics: field accuracy, document accuracy, response completeness
4. Generate evaluation summary with detailed performance breakdown
5. Export evaluation results to CSV for further analysis

Accuracy Calculation Methods:
- Financial fields (GST, TOTAL, SUBTOTAL): Numeric comparison with 0.01 tolerance
- List fields (QUANTITIES, PRICES, DESCRIPTIONS): Pipe-separated exact matching
- Date fields: Flexible format comparison with standardization
- String fields: Fuzzy matching (exact=1.0, partial=0.8, none=0.0)
- Bank fields: Exact matching for BSB, account numbers
"""

print("🎯 Starting Ground Truth Evaluation Against InternVL3 Results...")
print("=" * 70)

try:
    # Load extraction results
    extraction_csv_path = Path(output_dir) / "internvl3_batch_extraction.csv"

    if not extraction_csv_path.exists():
        print("❌ Extraction results CSV not found. Please run batch processing first.")
    else:
        print("📊 Loading extraction results and ground truth data...")

        # Load extraction results
        extraction_df = pd.read_csv(extraction_csv_path)
        print(f"✅ Loaded extraction results: {len(extraction_df)} images")

        # Load ground truth data
        if ground_truth_data:
            print(f"✅ Using ground truth data: {len(ground_truth_data)} images")

            # Perform comprehensive evaluation
            evaluation_results = []
            field_accuracy_totals = {field: [] for field in EXTRACTION_FIELDS}
            overall_accuracies = []

            print("\n🔍 Evaluating extraction accuracy for each image...")

            for idx, row in extraction_df.iterrows():
                image_name = row["image_name"]
                print(f"📊 Evaluating ({idx + 1}/{len(extraction_df)}): {image_name}")

                # Get ground truth for this image
                gt_data = ground_truth_data.get(image_name, {})

                if not gt_data:
                    print(f"   ⚠️ No ground truth found for {image_name} - skipping")
                    continue

                # Calculate field-wise accuracies
                image_evaluation = {
                    "image_name": image_name,
                    "extraction_results": {},
                    "ground_truth": {},
                    "field_accuracies": {},
                    "accuracy_details": {},
                }

                correct_fields = 0
                total_fields = 0

                for field in EXTRACTION_FIELDS:
                    extracted_value = str(row.get(field, "N/A"))
                    gt_value = str(gt_data.get(field, "N/A"))

                    # Calculate field accuracy using our sophisticated comparison
                    field_accuracy = calculate_field_accuracy(
                        extracted_value, gt_value, field
                    )

                    # Store evaluation data
                    image_evaluation["extraction_results"][field] = extracted_value
                    image_evaluation["ground_truth"][field] = gt_value
                    image_evaluation["field_accuracies"][field] = field_accuracy

                    # Track field accuracy for global statistics
                    field_accuracy_totals[field].append(field_accuracy)

                    # Count correct fields
                    if field_accuracy >= 0.99:  # Consider >= 99% as correct
                        correct_fields += 1
                    total_fields += 1

                    # Store accuracy details for debugging
                    if field_accuracy < 1.0:
                        image_evaluation["accuracy_details"][field] = {
                            "extracted": extracted_value,
                            "ground_truth": gt_value,
                            "accuracy": field_accuracy,
                            "comparison_type": "numeric"
                            if field
                            in [
                                "GST",
                                "TOTAL",
                                "SUBTOTAL",
                                "OPENING_BALANCE",
                                "CLOSING_BALANCE",
                            ]
                            else "list"
                            if field in ["QUANTITIES", "PRICES", "DESCRIPTIONS"]
                            else "string",
                        }

                # Calculate overall accuracy for this image
                image_accuracy = sum(
                    image_evaluation["field_accuracies"].values()
                ) / len(image_evaluation["field_accuracies"])
                image_evaluation["overall_accuracy"] = image_accuracy
                image_evaluation["correct_fields"] = correct_fields
                image_evaluation["total_fields"] = total_fields
                image_evaluation["field_accuracy_rate"] = (
                    correct_fields / total_fields
                ) * 100

                evaluation_results.append(image_evaluation)
                overall_accuracies.append(image_accuracy)

                # Show progress
                print(
                    f"   ✅ {correct_fields}/{total_fields} fields correct ({image_accuracy:.1%} accuracy)"
                )

                # Show problem fields if any
                problem_fields = [
                    f
                    for f, acc in image_evaluation["field_accuracies"].items()
                    if acc < 0.99
                ]
                if (
                    problem_fields and len(problem_fields) <= 5
                ):  # Only show if few problems
                    print(
                        f"   🔍 Fields needing attention: {', '.join(problem_fields)}"
                    )

            # Calculate comprehensive evaluation metrics
            print("\n📈 Calculating Comprehensive Evaluation Metrics...")

            total_images_evaluated = len(evaluation_results)
            overall_accuracy = (
                sum(overall_accuracies) / len(overall_accuracies)
                if overall_accuracies
                else 0.0
            )

            # Field-wise accuracy analysis
            field_accuracies = {}
            for field, accuracies in field_accuracy_totals.items():
                field_accuracies[field] = (
                    sum(accuracies) / len(accuracies) if accuracies else 0.0
                )

            # Document-level analysis
            perfect_documents = sum(1 for acc in overall_accuracies if acc >= 0.99)
            good_documents = sum(1 for acc in overall_accuracies if 0.8 <= acc < 0.99)
            fair_documents = sum(1 for acc in overall_accuracies if 0.6 <= acc < 0.8)
            poor_documents = sum(1 for acc in overall_accuracies if acc < 0.6)

            # Generate comprehensive evaluation report
            print("\n" + "=" * 70)
            print("📊 COMPREHENSIVE GROUND TRUTH EVALUATION RESULTS")
            print("=" * 70)

            print("\n🎯 Overall Performance:")
            print(f"   • Images evaluated: {total_images_evaluated}")
            print(f"   • Average accuracy: {overall_accuracy:.1%}")
            print(
                f"   • Perfect extractions (≥99%): {perfect_documents} ({perfect_documents / total_images_evaluated * 100:.1f}%)"
            )
            print(
                f"   • Good extractions (80-98%): {good_documents} ({good_documents / total_images_evaluated * 100:.1f}%)"
            )
            print(
                f"   • Fair extractions (60-79%): {fair_documents} ({fair_documents / total_images_evaluated * 100:.1f}%)"
            )
            print(
                f"   • Poor extractions (<60%): {poor_documents} ({poor_documents / total_images_evaluated * 100:.1f}%)"
            )

            # Field-level performance analysis
            sorted_field_accuracies = sorted(
                field_accuracies.items(), key=lambda x: x[1], reverse=True
            )

            print("\n📈 Top 10 Most Accurate Fields:")
            for i, (field, accuracy) in enumerate(sorted_field_accuracies[:10], 1):
                print(f"   {i:2d}. {field:<20} {accuracy:.1%}")

            print("\n📉 Bottom 10 Least Accurate Fields:")
            for i, (field, accuracy) in enumerate(
                sorted_field_accuracies[-10:], len(sorted_field_accuracies) - 9
            ):
                print(f"   {i:2d}. {field:<20} {accuracy:.1%}")

            # Identify best and worst performing images
            if evaluation_results:
                best_result = max(
                    evaluation_results, key=lambda x: x["overall_accuracy"]
                )
                worst_result = min(
                    evaluation_results, key=lambda x: x["overall_accuracy"]
                )

                print(
                    f"\n🏆 Best Performance: {best_result['image_name']} ({best_result['overall_accuracy']:.1%})"
                )
                print(
                    f"🔧 Worst Performance: {worst_result['image_name']} ({worst_result['overall_accuracy']:.1%})"
                )

                # Show problem analysis for worst performing image
                if worst_result["accuracy_details"]:
                    print(f"\n🔍 Problem Analysis for {worst_result['image_name']}:")
                    for field, details in list(
                        worst_result["accuracy_details"].items()
                    )[:5]:  # Show top 5 problems
                        print(f"   • {field}: {details['accuracy']:.1%} accuracy")
                        print(
                            f"     Extracted: '{details['extracted'][:50]}...' if len(details['extracted']) > 50 else details['extracted']"
                        )
                        print(
                            f"     Expected:  '{details['ground_truth'][:50]}...' if len(details['ground_truth']) > 50 else details['ground_truth']"
                        )

            # Export detailed evaluation results to CSV
            print("\n💾 Exporting Detailed Evaluation Results...")

            # Create evaluation summary DataFrame
            evaluation_summary_data = []
            for result in evaluation_results:
                row_data = {
                    "image_name": result["image_name"],
                    "overall_accuracy": result["overall_accuracy"],
                    "correct_fields": result["correct_fields"],
                    "total_fields": result["total_fields"],
                    "field_accuracy_rate": result["field_accuracy_rate"],
                }

                # Add field-wise accuracies
                for field in EXTRACTION_FIELDS:
                    row_data[f"{field}_accuracy"] = result["field_accuracies"].get(
                        field, 0.0
                    )
                    row_data[f"{field}_extracted"] = result["extraction_results"].get(
                        field, "N/A"
                    )
                    row_data[f"{field}_ground_truth"] = result["ground_truth"].get(
                        field, "N/A"
                    )

                evaluation_summary_data.append(row_data)

            evaluation_df = pd.DataFrame(evaluation_summary_data)

            # Save evaluation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_csv_path = (
                Path(output_dir) / f"internvl3_ground_truth_evaluation_{timestamp}.csv"
            )

            eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
            evaluation_df.to_csv(eval_csv_path, index=False, encoding="utf-8")

            print(f"✅ Detailed evaluation saved: {eval_csv_path}")
            print(
                f"📊 Evaluation CSV contains: {len(evaluation_df)} rows × {len(evaluation_df.columns)} columns"
            )

            # Save summary statistics
            summary_stats = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_images_evaluated": total_images_evaluated,
                "overall_accuracy": overall_accuracy,
                "perfect_documents": perfect_documents,
                "good_documents": good_documents,
                "fair_documents": fair_documents,
                "poor_documents": poor_documents,
                "field_accuracies": field_accuracies,
                "best_performing_image": best_result["image_name"]
                if evaluation_results
                else "N/A",
                "best_performance_accuracy": best_result["overall_accuracy"]
                if evaluation_results
                else 0.0,
                "worst_performing_image": worst_result["image_name"]
                if evaluation_results
                else "N/A",
                "worst_performance_accuracy": worst_result["overall_accuracy"]
                if evaluation_results
                else 0.0,
            }

            stats_json_path = (
                Path(output_dir) / f"internvl3_evaluation_summary_{timestamp}.json"
            )
            with stats_json_path.open("w", encoding="utf-8") as f:
                json.dump(summary_stats, f, indent=2, default=str)

            print(f"📈 Summary statistics saved: {stats_json_path}")

            print("\n✅ Ground Truth Evaluation Completed Successfully!")
            print(
                f"🎯 InternVL3 achieved {overall_accuracy:.1%} average accuracy on the evaluation dataset"
            )
            print(f"📁 Results saved to: {output_dir}")

        else:
            print(
                "❌ Ground truth data not available. Please ensure ground truth loading completed successfully."
            )

except Exception as e:
    print(f"❌ Error during ground truth evaluation: {e}")
    print(f"🔍 Error type: {type(e).__name__}")

    import traceback

    print("\n🔧 Full error traceback:")
    traceback.print_exc()


"""
Comprehensive Evaluation Summary Report Generator

Purpose:
- Generate final comprehensive evaluation report combining all metrics
- Create executive summary with key findings and recommendations
- Export consolidated evaluation results for stakeholder review
- Provide actionable insights for model deployment decisions

Report Components:
1. Executive Summary with key performance indicators
2. Detailed accuracy breakdown by field type and document complexity
3. Model performance analysis with strengths and weaknesses
4. Comparative analysis against expected benchmarks
5. Deployment readiness assessment and recommendations
6. Technical appendix with detailed metrics
"""

print("📊 GENERATING COMPREHENSIVE EVALUATION SUMMARY REPORT")
print("=" * 80)

try:
    # Check if evaluation has been run
    output_dir_path = Path(output_dir)
    evaluation_files = list(
        output_dir_path.glob("internvl3_ground_truth_evaluation_*.csv")
    )
    summary_files = list(output_dir_path.glob("internvl3_evaluation_summary_*.json"))

    if not evaluation_files or not summary_files:
        print(
            "❌ Evaluation results not found. Please run the ground truth evaluation cell first."
        )
        print("💡 Expected files:")
        print("   • internvl3_ground_truth_evaluation_*.csv")
        print("   • internvl3_evaluation_summary_*.json")
    else:
        # Load the most recent evaluation results
        latest_eval_file = max(evaluation_files, key=lambda p: p.stat().st_mtime)
        latest_summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)

        print(f"📈 Loading evaluation results from: {latest_eval_file.name}")
        print(f"📊 Loading summary statistics from: {latest_summary_file.name}")

        # Load evaluation data
        eval_df = pd.read_csv(latest_eval_file)

        with latest_summary_file.open("r", encoding="utf-8") as f:
            summary_stats = json.load(f)

        print(f"✅ Loaded evaluation data: {len(eval_df)} images evaluated")

        # Generate executive summary report
        report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        executive_summary = f"""
# InternVL3-2B Key-Value Extraction Evaluation Report
## Executive Summary

**Generated:** {report_timestamp}
**Evaluation Dataset:** 20 synthetic business documents
**Model:** InternVL3-2B Vision-Language Model
**Task:** Structured key-value extraction from business documents

### 🎯 Overall Performance Summary

**Average Accuracy:** {summary_stats["overall_accuracy"]:.1%}
- Perfect Extractions (≥99%): {summary_stats["perfect_documents"]}/{summary_stats["total_images_evaluated"]} ({summary_stats["perfect_documents"] / summary_stats["total_images_evaluated"] * 100:.1f}%)
- Good Extractions (80-98%): {summary_stats["good_documents"]}/{summary_stats["total_images_evaluated"]} ({summary_stats["good_documents"] / summary_stats["total_images_evaluated"] * 100:.1f}%)
- Fair Extractions (60-79%): {summary_stats["fair_documents"]}/{summary_stats["total_images_evaluated"]} ({summary_stats["fair_documents"] / summary_stats["total_images_evaluated"] * 100:.1f}%)
- Poor Extractions (<60%): {summary_stats["poor_documents"]}/{summary_stats["total_images_evaluated"]} ({summary_stats["poor_documents"] / summary_stats["total_images_evaluated"] * 100:.1f}%)

### 📊 Key Findings

1. **Model Reliability:** InternVL3-2B demonstrates consistent performance across diverse document types
2. **Field Extraction:** Successfully extracts {len([f for f, acc in summary_stats["field_accuracies"].items() if acc >= 0.9])} out of 25 fields with ≥90% accuracy
3. **Best Performance:** {summary_stats["best_performing_image"]} ({summary_stats["best_performance_accuracy"]:.1%} accuracy)
4. **Challenging Cases:** {summary_stats["worst_performing_image"]} ({summary_stats["worst_performance_accuracy"]:.1%} accuracy)

### 🏆 Top Performing Fields
"""

        # Add top performing fields
        sorted_fields = sorted(
            summary_stats["field_accuracies"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (field, accuracy) in enumerate(sorted_fields[:10], 1):
            executive_summary += f"{i:2d}. {field:<20} {accuracy:.1%}\n"

        executive_summary += f"""
### 📈 Deployment Readiness Assessment

**Overall Grade:** {"A+ (Excellent)" if summary_stats["overall_accuracy"] >= 0.9 else "A (Good)" if summary_stats["overall_accuracy"] >= 0.8 else "B (Fair)" if summary_stats["overall_accuracy"] >= 0.7 else "C (Needs Improvement)"}

**Recommendations:**
"""

        if summary_stats["overall_accuracy"] >= 0.9:
            executive_summary += """- ✅ **READY FOR PRODUCTION:** Model demonstrates excellent accuracy and consistency
- ✅ **HIGH RELIABILITY:** Suitable for automated document processing workflows
- ✅ **MINIMAL OVERSIGHT:** Requires minimal human validation in production"""
        elif summary_stats["overall_accuracy"] >= 0.8:
            executive_summary += """- ✅ **READY FOR PRODUCTION:** Model shows good performance with minor limitations
- ⚠️ **MODERATE OVERSIGHT:** Recommend validation for critical business fields
- ✅ **SUITABLE FOR AUTOMATION:** Can handle majority of documents automatically"""
        elif summary_stats["overall_accuracy"] >= 0.7:
            executive_summary += """- ⚠️ **REQUIRES OPTIMIZATION:** Consider fine-tuning or prompt engineering
- ⚠️ **INCREASED OVERSIGHT:** Human validation recommended for important documents
- 🔧 **PILOT DEPLOYMENT:** Suitable for pilot programs with close monitoring"""
        else:
            executive_summary += """- ❌ **NOT READY FOR PRODUCTION:** Significant accuracy improvements needed
- 🔧 **REQUIRES INVESTIGATION:** Review model configuration and training data
- 📋 **HUMAN VALIDATION REQUIRED:** Manual review necessary for all extractions"""

        executive_summary += """

### 📋 Technical Details
- **Extraction Fields:** 25 structured business document fields
- **Document Types:** Invoices, receipts, bank statements, tax documents
- **Evaluation Method:** Sophisticated field-specific comparison with tolerance for numeric fields
- **Data Quality:** Ground truth validated against 20 diverse synthetic business documents

---
*Report generated automatically by InternVL3 evaluation system*
"""

        print("📝 EXECUTIVE SUMMARY GENERATED")
        print("=" * 50)
        print(executive_summary)

        # Save comprehensive report
        report_filename = f"internvl3_comprehensive_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = output_dir_path / report_filename

        with report_path.open("w", encoding="utf-8") as f:
            f.write(executive_summary)

        print(f"\n💾 Comprehensive report saved: {report_path}")

        # Generate detailed field analysis
        print("\n🔍 DETAILED FIELD ANALYSIS")
        print("=" * 50)

        # Categorize fields by performance
        excellent_fields = [
            f for f, acc in summary_stats["field_accuracies"].items() if acc >= 0.95
        ]
        good_fields = [
            f
            for f, acc in summary_stats["field_accuracies"].items()
            if 0.8 <= acc < 0.95
        ]
        challenging_fields = [
            f for f, acc in summary_stats["field_accuracies"].items() if acc < 0.8
        ]

        print(f"📈 Excellent Performance (≥95%): {len(excellent_fields)} fields")
        if excellent_fields:
            for field in excellent_fields[:5]:  # Show top 5
                acc = summary_stats["field_accuracies"][field]
                print(f"   ✅ {field}: {acc:.1%}")
            if len(excellent_fields) > 5:
                print(f"   ... and {len(excellent_fields) - 5} more")

        print(f"\n📊 Good Performance (80-94%): {len(good_fields)} fields")
        if good_fields:
            for field in good_fields[:5]:  # Show top 5
                acc = summary_stats["field_accuracies"][field]
                print(f"   ✅ {field}: {acc:.1%}")
            if len(good_fields) > 5:
                print(f"   ... and {len(good_fields) - 5} more")

        print(f"\n🔧 Challenging Fields (<80%): {len(challenging_fields)} fields")
        if challenging_fields:
            for field in challenging_fields:
                acc = summary_stats["field_accuracies"][field]
                print(f"   ⚠️ {field}: {acc:.1%}")

        # Generate deployment checklist
        deployment_checklist = f"""
# InternVL3-2B Deployment Checklist

## ✅ Pre-Deployment Validation
- [{"x" if summary_stats["overall_accuracy"] >= 0.8 else " "}] Overall accuracy ≥80% ({summary_stats["overall_accuracy"]:.1%})
- [{"x" if summary_stats["perfect_documents"] / summary_stats["total_images_evaluated"] >= 0.7 else " "}] Perfect extractions ≥70% ({summary_stats["perfect_documents"] / summary_stats["total_images_evaluated"] * 100:.1f}%)
- [{"x" if len(excellent_fields) >= 15 else " "}] Excellent fields ≥15 ({len(excellent_fields)})
- [{"x" if len(challenging_fields) <= 5 else " "}] Challenging fields ≤5 ({len(challenging_fields)})

## 🎯 Production Readiness
- Model: InternVL3-2B Vision-Language Model
- Evaluation: {summary_stats["total_images_evaluated"]} documents tested
- Best Case: {summary_stats["best_performance_accuracy"]:.1%} accuracy
- Worst Case: {summary_stats["worst_performance_accuracy"]:.1%} accuracy

## 📊 Monitoring Recommendations
- Track accuracy for critical fields: {", ".join(excellent_fields[:5])}
- Monitor challenging fields: {", ".join(challenging_fields) if challenging_fields else "None"}
- Implement validation for financial fields (GST, TOTAL, SUBTOTAL)
- Regular evaluation against new document types

## 🚀 Next Steps
{"1. ✅ DEPLOY TO PRODUCTION - Model ready for automated processing" if summary_stats["overall_accuracy"] >= 0.9 else "1. ⚠️ PILOT DEPLOYMENT - Test with subset of documents" if summary_stats["overall_accuracy"] >= 0.8 else "1. 🔧 OPTIMIZATION REQUIRED - Improve model before deployment"}
2. 📋 Establish monitoring dashboards for accuracy tracking
3. 🔄 Plan regular model evaluation and updates
4. 📚 Document operational procedures and fallback processes
"""

        checklist_path = (
            output_dir_path
            / f"internvl3_deployment_checklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with checklist_path.open("w", encoding="utf-8") as f:
            f.write(deployment_checklist)

        print("\n📋 DEPLOYMENT CHECKLIST")
        print("=" * 30)
        print(deployment_checklist)

        print(f"\n💾 Deployment checklist saved: {checklist_path}")

        # Final summary
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(
            f"📊 InternVL3-2B achieved {summary_stats['overall_accuracy']:.1%} average accuracy"
        )
        print(
            f"📈 {summary_stats['perfect_documents']} out of {summary_stats['total_images_evaluated']} documents had perfect extraction"
        )
        print(
            f"🎯 {len(excellent_fields)} out of 25 fields achieved excellent performance (≥95%)"
        )
        print(f"📁 All results saved to: {output_dir}")
        print("\n📋 Generated Files:")
        print(f"   • {latest_eval_file.name} - Detailed evaluation results")
        print(f"   • {latest_summary_file.name} - Summary statistics")
        print(f"   • {report_filename} - Executive summary report")
        print(f"   • {checklist_path.name} - Deployment checklist")

except Exception as e:
    print(f"❌ Error generating comprehensive report: {e}")
    print(f"🔍 Error type: {type(e).__name__}")

    import traceback

    print("\n🔧 Full error traceback:")
    traceback.print_exc()
