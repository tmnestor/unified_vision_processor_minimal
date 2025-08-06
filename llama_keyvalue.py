#!/usr/bin/env python3
"""
Llama Vision Key-Value Extraction with Comprehensive Evaluation

Purpose:
- Load Llama-3.2-11B-Vision-Instruct model for structured document analysis
- Execute comprehensive evaluation pipeline using InternVL3's sophisticated infrastructure
- Generate detailed reports and deployment readiness assessments
- Provide direct comparison capabilities with other vision models

Key Features:
- Advanced batch processing with multimodal conversation format
- Sophisticated field-specific accuracy calculation with fixed N/A handling
- Comprehensive evaluation metrics and quality assessment
- Executive summary generation and deployment checklists
- Production readiness indicators and optimization recommendations

Architecture:
- Adapts Llama's multimodal message structure for document processing
- Ports InternVL3's evaluation infrastructure with accuracy improvements
- Maintains compatibility with existing ground truth and output formats
- Supports 25-field structured extraction for business documents
"""

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND GLOBAL VARIABLES
# ============================================================================
data_dir = "/home/jovyan/nfs_share/tod/evaluation_data"  # 20 test images
ground_truth_path = "/home/jovyan/nfs_share/tod/unified_vision_processor_minimal/evaluation_ground_truth.csv"  # Ground truth CSV
model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct" 
output_dir = "/home/jovyan/nfs_share/tod/output"

# 25 extraction fields in alphabetical order for consistency with InternVL3
EXTRACTION_FIELDS = [
    'ABN', 'ACCOUNT_HOLDER', 'BANK_ACCOUNT_NUMBER', 'BANK_NAME', 'BSB_NUMBER',
    'BUSINESS_ADDRESS', 'BUSINESS_PHONE', 'CLOSING_BALANCE', 'DESCRIPTIONS',
    'DOCUMENT_TYPE', 'DUE_DATE', 'GST', 'INVOICE_DATE', 'OPENING_BALANCE',
    'PAYER_ADDRESS', 'PAYER_EMAIL', 'PAYER_NAME', 'PAYER_PHONE', 'PRICES',
    'QUANTITIES', 'STATEMENT_PERIOD', 'SUBTOTAL', 'SUPPLIER', 'SUPPLIER_WEBSITE', 'TOTAL'
]

# Enhanced extraction prompt optimized for Llama Vision with strict formatting
EXTRACTION_PROMPT = """Extract key-value data from this business document image.

CRITICAL INSTRUCTIONS:
- Output ONLY the structured data below
- Do NOT include any conversation text
- Do NOT repeat the user's request
- Do NOT include <image> tokens
- Start immediately with DOCUMENT_TYPE
- Stop immediately after DESCRIPTIONS

REQUIRED OUTPUT FORMAT - EXACTLY 25 LINES:
DOCUMENT_TYPE: [value or N/A]
SUPPLIER: [value or N/A]
ABN: [11-digit Australian Business Number or N/A]
PAYER_NAME: [value or N/A]
PAYER_ADDRESS: [value or N/A]
PAYER_PHONE: [value or N/A]
PAYER_EMAIL: [value or N/A]
INVOICE_DATE: [value or N/A]
DUE_DATE: [value or N/A]
GST: [GST amount in dollars or N/A]
TOTAL: [total amount in dollars or N/A]
SUBTOTAL: [subtotal amount in dollars or N/A]
SUPPLIER_WEBSITE: [value or N/A]
QUANTITIES: [list of quantities or N/A]
PRICES: [individual prices in dollars or N/A]
BUSINESS_ADDRESS: [value or N/A]
BUSINESS_PHONE: [value or N/A]
BANK_NAME: [bank name from bank statements only or N/A]
BSB_NUMBER: [6-digit BSB from bank statements only or N/A]
BANK_ACCOUNT_NUMBER: [account number from bank statements only or N/A]
ACCOUNT_HOLDER: [value or N/A]
STATEMENT_PERIOD: [value or N/A]
OPENING_BALANCE: [opening balance amount in dollars or N/A]
CLOSING_BALANCE: [closing balance amount in dollars or N/A]
DESCRIPTIONS: [list of transaction descriptions or N/A]

FORMAT RULES:
- Use exactly: KEY: value (colon and space)
- NEVER use: **KEY:** or **KEY** or *KEY* or any formatting
- Plain text only - NO markdown, NO bold, NO italic
- Include ALL 25 keys even if value is N/A
- Output ONLY these 25 lines, nothing else

STOP after DESCRIPTIONS line. Do not add explanations or comments."""

print("🦙 Llama Vision Key-Value Extraction with Comprehensive Evaluation")
print(f"📁 Data directory: {data_dir}")
print(f"📂 Output directory: {output_dir}")
print(f"📊 Ground truth: {ground_truth_path}")
print(f"🔧 Model: {model_path}")

# ============================================================================
# MODEL LOADING AND INITIALIZATION
# ============================================================================

def load_llama_model():
    """
    Load Llama-3.2-11B-Vision-Instruct model with optimal configuration
    
    Returns:
        tuple: (model, processor) for document processing
    """
    print(f"\n🔄 Loading Llama Vision model from: {model_path}")
    
    try:
        # Load model with optimal configuration
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Memory-efficient 16-bit precision
            device_map="auto",           # Automatic device mapping
        )
        
        # Load processor for multimodal inputs
        processor = AutoProcessor.from_pretrained(model_path)
        
        print("✅ Llama Vision model loaded successfully")
        print(f"🔧 Device: {model.device}")
        print(f"💾 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading Llama model: {e}")
        raise

# Initialize global model and processor
print("🚀 Initializing Llama Vision model...")
model, processor = load_llama_model()

# ============================================================================
# IMAGE DISCOVERY AND LOADING
# ============================================================================

def discover_images(directory_path):
    """
    Discover all image files in the specified directory
    
    Args:
        directory_path (str): Path to directory containing images
        
    Returns:
        list: List of image file paths found in directory
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(str(p) for p in Path(directory_path).glob(extension))
    
    # Sort for consistent processing order
    image_files.sort()
    return image_files

def load_document_image(image_path):
    """
    Load document image with error handling
    
    Args:
        image_path (str): Path to document image
        
    Returns:
        PIL.Image: Loaded document image
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        raise

# ============================================================================
# RESPONSE PARSING WITH LLAMA-SPECIFIC HANDLING
# ============================================================================

def parse_extraction_response(response_text):
    """
    Parse Llama extraction response into structured dictionary with success tracking
    
    Args:
        response_text (str): Raw text response from Llama extraction
        
    Returns:
        tuple: (field_dict, extracted_fields_set, success_metadata)
    """
    field_dict = {}
    extracted_fields = set()
    
    # Clean Llama conversation artifacts
    lines = response_text.split('\n')
    cleaned_lines = []
    in_response = False
    
    for line in lines:
        line = line.strip()
        # Look for start of structured output
        if line.startswith('DOCUMENT_TYPE:') or any(line.startswith(f'{field}:') for field in EXTRACTION_FIELDS):
            in_response = True
        # Skip conversation artifacts
        if line.startswith(('user', 'assistant', '<image>', 'Extract data')):
            in_response = False
            continue
        # Collect structured response lines
        if in_response and ':' in line and not line.startswith('<'):
            # Remove markdown artifacts common with Llama
            clean_line = line.replace('**', '').replace('*', '').strip()
            cleaned_lines.append(clean_line)
    
    # Parse cleaned lines
    for line in cleaned_lines:
        if ':' in line:
            try:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                # Only process expected fields
                if key in EXTRACTION_FIELDS:
                    field_dict[key] = value if value else 'N/A'
                    extracted_fields.add(key)
                    
            except ValueError:
                continue
    
    # Fill missing fields
    for field in EXTRACTION_FIELDS:
        if field not in extracted_fields:
            field_dict[field] = 'N/A'
    
    # Calculate success metadata
    successful_extractions = len(extracted_fields)
    fields_with_content = len([f for f in extracted_fields if field_dict[f] != 'N/A'])
    
    success_metadata = {
        'response_completeness': successful_extractions,
        'response_completeness_rate': (successful_extractions / len(EXTRACTION_FIELDS)) * 100,
        'content_coverage': fields_with_content,
        'content_coverage_rate': (fields_with_content / successful_extractions) * 100 if successful_extractions > 0 else 0,
        'failed_extractions': len(EXTRACTION_FIELDS) - successful_extractions
    }
    
    return field_dict, extracted_fields, success_metadata

# ============================================================================
# BATCH PROCESSING WITH LLAMA MULTIMODAL FORMAT
# ============================================================================

def process_image_batch(image_files, progress_callback=None):
    """
    Process batch of images through Llama Vision extraction pipeline
    
    Args:
        image_files (list): List of image file paths
        progress_callback (callable, optional): Progress update function
        
    Returns:
        tuple: (results, batch_statistics)
    """
    results = []
    batch_stats = {
        'total_images': len(image_files),
        'successful_responses': 0,
        'total_fields_returned': 0,
        'total_fields_with_content': 0,
        'processing_errors': 0
    }
    
    print(f"🚀 Starting Llama Vision batch processing of {len(image_files)} images...")
    
    for i, image_file in enumerate(image_files, 1):
        image_name = Path(image_file).name
        print(f"📷 Processing ({i}/{len(image_files)}): {image_name}")
        
        try:
            # Load image
            image = load_document_image(image_file)
            
            # Create Llama's multimodal message structure
            message_structure = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": EXTRACTION_PROMPT}
                    ]
                }
            ]
            
            # Apply chat template
            text_input = processor.apply_chat_template(
                message_structure, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = processor(image, text_input, return_tensors="pt").to(model.device)
            
            # Generate response
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            
            # Decode response
            response = processor.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            
            # Parse response
            extracted_fields, returned_fields, success_meta = parse_extraction_response(response)
            
            # Create result
            result_row = {'image_name': image_name}
            result_row.update(extracted_fields)
            result_row['_response_completeness'] = success_meta['response_completeness']
            result_row['_content_coverage'] = success_meta['content_coverage']
            results.append(result_row)
            
            # Update statistics
            batch_stats['successful_responses'] += 1
            batch_stats['total_fields_returned'] += success_meta['response_completeness']
            batch_stats['total_fields_with_content'] += success_meta['content_coverage']
            
            print(f"   ✅ Model returned {success_meta['response_completeness']}/25 fields ({success_meta['response_completeness_rate']:.1f}%)")
            print(f"   📊 Content in {success_meta['content_coverage']} fields ({success_meta['content_coverage_rate']:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Processing error for {image_name}: {str(e)}")
            
            # Create error result
            error_result = {'image_name': image_name}
            error_result.update({field: 'N/A' for field in EXTRACTION_FIELDS})
            error_result['_response_completeness'] = 0
            error_result['_content_coverage'] = 0
            results.append(error_result)
            
            batch_stats['processing_errors'] += 1
        
        if progress_callback:
            progress_callback(i, len(image_files), image_name)
    
    return results, batch_stats

def create_extraction_dataframe(results):
    """
    Create pandas DataFrame from extraction results
    
    Args:
        results (list): Extraction results
        
    Returns:
        tuple: (main_df, metadata_df)
    """
    if not results:
        columns = ['image_name'] + EXTRACTION_FIELDS
        return pd.DataFrame(columns=columns), pd.DataFrame()
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Extract metadata
    metadata_columns = ['image_name', '_response_completeness', '_content_coverage']
    metadata_df = results_df[metadata_columns].copy() if all(col in results_df.columns for col in metadata_columns) else pd.DataFrame()
    
    # Main DataFrame with proper column ordering
    main_columns = ['image_name'] + EXTRACTION_FIELDS
    main_df = results_df[main_columns] if all(col in results_df.columns for col in main_columns) else results_df
    
    column_order = ['image_name'] + EXTRACTION_FIELDS
    main_df = main_df.reindex(columns=column_order, fill_value='N/A')
    
    return main_df, metadata_df

# ============================================================================
# GROUND TRUTH LOADING AND VALIDATION
# ============================================================================

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
        print(f"📊 Loaded ground truth: {len(gt_df)} rows × {len(gt_df.columns)} columns")
        
        # Validate structure
        expected_columns = ['image_file'] + EXTRACTION_FIELDS
        actual_columns = list(gt_df.columns)
        
        if len(actual_columns) != len(expected_columns):
            print(f"⚠️ Column count mismatch: expected {len(expected_columns)}, got {len(actual_columns)}")
        
        # Check field alignment
        missing_fields = set(expected_columns) - set(actual_columns)
        extra_fields = set(actual_columns) - set(expected_columns)
        
        if missing_fields:
            print(f"⚠️ Missing fields in ground truth: {missing_fields}")
        if extra_fields:
            print(f"⚠️ Extra fields in ground truth: {extra_fields}")
        
        # Create mapping
        ground_truth_map = {}
        for _, row in gt_df.iterrows():
            image_file = row['image_file']
            gt_data = {field: str(row[field]) if pd.notna(row[field]) else 'N/A' 
                      for field in EXTRACTION_FIELDS if field in row.index}
            ground_truth_map[image_file] = gt_data
        
        print(f"✅ Created ground truth mapping for {len(ground_truth_map)} images")
        return ground_truth_map
        
    except FileNotFoundError:
        print(f"❌ Ground truth file not found: {csv_path}")
        return {}
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return {}

# ============================================================================
# FIELD ACCURACY CALCULATION WITH FIXED N/A HANDLING  
# ============================================================================

def calculate_field_accuracy(extracted_value, ground_truth_value, field_name):
    """
    Calculate accuracy for a specific field using sophisticated comparison logic
    FIXED: Now includes comprehensive N/A variants including 'nan'
    
    Args:
        extracted_value (str): Value extracted by the model
        ground_truth_value (str): Correct value from ground truth
        field_name (str): Name of the field being compared
        
    Returns:
        float: Accuracy score between 0.0 and 1.0
    """
    # FIXED: Comprehensive N/A variants including 'nan' which was missing
    na_variants = ['N/A', 'NA', '', 'NAN', 'NULL', 'NONE', 'NIL']
    
    # Handle missing values with expanded variant detection
    if not ground_truth_value or str(ground_truth_value).upper() in na_variants:
        return 1.0 if (not extracted_value or str(extracted_value).upper() in na_variants) else 0.0
    
    if not extracted_value or str(extracted_value).upper() in na_variants:
        return 0.0  # Ground truth exists but nothing extracted
    
    # Normalize for comparison
    extracted_clean = str(extracted_value).strip()
    gt_clean = str(ground_truth_value).strip()
    
    # Exact match (case-insensitive)
    if extracted_clean.lower() == gt_clean.lower():
        return 1.0
    
    # Field-specific comparison logic
    if field_name in ['GST', 'TOTAL', 'SUBTOTAL', 'OPENING_BALANCE', 'CLOSING_BALANCE']:
        # Numeric comparison with tolerance for financial fields
        try:
            ext_num = float(re.sub(r'[^\d.-]', '', extracted_clean.replace(',', '')))
            gt_num = float(re.sub(r'[^\d.-]', '', gt_clean.replace(',', '')))
            
            tolerance = 0.01
            return 1.0 if abs(ext_num - gt_num) < tolerance else 0.0
            
        except (ValueError, TypeError):
            return 1.0 if extracted_clean.lower() == gt_clean.lower() else 0.0
    
    elif field_name in ['QUANTITIES', 'PRICES', 'DESCRIPTIONS']:
        # List comparison for pipe-separated values
        try:
            ext_items = [item.strip() for item in extracted_clean.split('|')]
            gt_items = [item.strip() for item in gt_clean.split('|')]
            
            if len(ext_items) != len(gt_items):
                return 0.0
            
            matches = sum(1 for e, g in zip(ext_items, gt_items, strict=False) 
                         if e.lower().strip() == g.lower().strip())
            
            return matches / len(gt_items) if gt_items else 0.0
            
        except Exception:
            return 1.0 if extracted_clean.lower() == gt_clean.lower() else 0.0
    
    elif field_name in ['INVOICE_DATE', 'DUE_DATE']:
        # Date comparison with flexible format handling
        try:
            ext_date = re.sub(r'[^\d/\-]', '', extracted_clean)
            gt_date = re.sub(r'[^\d/\-]', '', gt_clean)
            
            return 1.0 if ext_date == gt_date else 0.0
            
        except Exception:
            return 1.0 if extracted_clean.lower() == gt_clean.lower() else 0.0
    
    else:
        # String comparison with fuzzy matching
        if extracted_clean.lower() == gt_clean.lower():
            return 1.0
        elif (extracted_clean.lower() in gt_clean.lower() or 
              gt_clean.lower() in extracted_clean.lower()):
            return 0.8  # Partial match
        else:
            return 0.0

# ============================================================================
# COMPREHENSIVE EVALUATION PIPELINE
# ============================================================================

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
    
    print(f"\n🎯 Evaluating {len(extraction_results)} Llama extraction results against ground truth...")
    
    for i, result in enumerate(extraction_results, 1):
        image_name = result['image_name']
        print(f"📊 Evaluating ({i}/{len(extraction_results)}): {image_name}")
        
        # Get ground truth
        gt_data = ground_truth_map.get(image_name, {})
        
        if not gt_data:
            print(f"   ⚠️ No ground truth found for {image_name}")
            continue
        
        # Calculate field-wise accuracies
        image_evaluation = {'image_name': image_name}
        image_field_accuracies = {}
        
        for field in EXTRACTION_FIELDS:
            extracted_value = result.get(field, 'N/A')
            gt_value = gt_data.get(field, 'N/A')
            
            accuracy = calculate_field_accuracy(extracted_value, gt_value, field)
            image_field_accuracies[field] = accuracy
            field_accuracies[field].append(accuracy)
            
            # Store evaluation data
            image_evaluation[f'{field}_extracted'] = extracted_value
            image_evaluation[f'{field}_ground_truth'] = gt_value
            image_evaluation[f'{field}_accuracy'] = accuracy
        
        # Calculate overall accuracy
        image_accuracy = sum(image_field_accuracies.values()) / len(image_field_accuracies)
        image_evaluation['overall_accuracy'] = image_accuracy
        overall_accuracies.append(image_accuracy)
        
        evaluation_data.append(image_evaluation)
        
        # Progress report
        fields_correct = sum(1 for acc in image_field_accuracies.values() if acc >= 0.99)
        print(f"   ✅ {fields_correct}/25 fields correct ({image_accuracy:.1%} accuracy)")
    
    # Calculate comprehensive metrics
    evaluation_summary = {
        'total_images': len(evaluation_data),
        'overall_accuracy': sum(overall_accuracies) / len(overall_accuracies) if overall_accuracies else 0.0,
        'perfect_documents': sum(1 for acc in overall_accuracies if acc >= 0.99),
        'good_documents': sum(1 for acc in overall_accuracies if 0.8 <= acc < 0.99),
        'fair_documents': sum(1 for acc in overall_accuracies if 0.6 <= acc < 0.8),
        'poor_documents': sum(1 for acc in overall_accuracies if acc < 0.6),
        'field_accuracies': {field: sum(accs) / len(accs) if accs else 0.0 
                           for field, accs in field_accuracies.items()},
        'evaluation_data': evaluation_data,
        'best_performing_image': max(evaluation_data, key=lambda x: x['overall_accuracy'])['image_name'] if evaluation_data else '',
        'worst_performing_image': min(evaluation_data, key=lambda x: x['overall_accuracy'])['image_name'] if evaluation_data else '',
        'best_performance_accuracy': max(overall_accuracies) if overall_accuracies else 0.0,
        'worst_performance_accuracy': min(overall_accuracies) if overall_accuracies else 0.0
    }
    
    return evaluation_summary

# ============================================================================
# REPORT GENERATION AND ANALYSIS
# ============================================================================

def generate_comprehensive_reports(evaluation_summary, output_dir_path):
    """
    Generate comprehensive evaluation reports including executive summary and deployment checklist
    
    Args:
        evaluation_summary (dict): Evaluation results and metrics
        output_dir_path (Path): Output directory path
        
    Returns:
        dict: Paths to generated reports
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Executive Summary Report
    summary_stats = evaluation_summary
    sorted_fields = sorted(summary_stats['field_accuracies'].items(), key=lambda x: x[1], reverse=True)
    
    executive_summary = f"""# Llama Vision Key-Value Extraction - Executive Summary

## Model Performance Overview
**Model:** Llama-3.2-11B-Vision-Instruct  
**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Documents Processed:** {summary_stats['total_images']}  
**Average Accuracy:** {summary_stats['overall_accuracy']:.1%}

## Key Findings

1. **Document Analysis:** Processed {summary_stats['total_images']} business documents with comprehensive field extraction
2. **Field Extraction:** Successfully extracts {len([f for f, acc in summary_stats['field_accuracies'].items() if acc >= 0.9])} out of 25 fields with ≥90% accuracy
3. **Best Performance:** {summary_stats['best_performing_image']} ({summary_stats['best_performance_accuracy']:.1%} accuracy)
4. **Challenging Cases:** {summary_stats['worst_performing_image']} ({summary_stats['worst_performance_accuracy']:.1%} accuracy)

## Field Performance Analysis

### Top Performing Fields (≥90% accuracy)
"""
    
    excellent_fields = [field for field, accuracy in sorted_fields if accuracy >= 0.9]
    if excellent_fields:
        for i, (field, accuracy) in enumerate([item for item in sorted_fields if item[1] >= 0.9][:10], 1):
            executive_summary += f"{i:2d}. {field:<20} {accuracy:.1%}\n"
    else:
        executive_summary += "No fields achieved ≥90% accuracy\n"
    
    executive_summary += """
### Challenging Fields (Requires Attention)
"""
    
    challenging_fields = [(field, accuracy) for field, accuracy in sorted_fields[-5:] if accuracy < 0.9]
    for i, (field, accuracy) in enumerate(challenging_fields, 1):
        executive_summary += f"{i}. {field:<20} {accuracy:.1%}\n"
    
    # Production readiness assessment
    if summary_stats['overall_accuracy'] >= 0.9:
        grade = "A+ (Excellent)"
        status = "✅ **READY FOR PRODUCTION:** Model demonstrates excellent accuracy and consistency"
    elif summary_stats['overall_accuracy'] >= 0.8:
        grade = "A (Good)" 
        status = "✅ **READY FOR PRODUCTION:** Model shows good performance with minor limitations"
    elif summary_stats['overall_accuracy'] >= 0.7:
        grade = "B (Fair)"
        status = "⚠️ **REQUIRES OPTIMIZATION:** Consider fine-tuning or prompt engineering"
    else:
        grade = "C (Needs Improvement)"
        status = "❌ **NOT READY FOR PRODUCTION:** Significant accuracy improvements needed"
    
    executive_summary += f"""
**Overall Grade:** {grade}

## Production Readiness Assessment

{status}

## Document Quality Distribution
- Perfect Documents (≥99%): {summary_stats['perfect_documents']} ({summary_stats['perfect_documents']/summary_stats['total_images']*100:.1f}%)
- Good Documents (80-98%): {summary_stats['good_documents']} ({summary_stats['good_documents']/summary_stats['total_images']*100:.1f}%)  
- Fair Documents (60-79%): {summary_stats['fair_documents']} ({summary_stats['fair_documents']/summary_stats['total_images']*100:.1f}%)
- Poor Documents (<60%): {summary_stats['poor_documents']} ({summary_stats['poor_documents']/summary_stats['total_images']*100:.1f}%)

## Recommendations

### Immediate Actions
{"1. ✅ DEPLOY TO PRODUCTION - Model ready for automated processing" if summary_stats['overall_accuracy'] >= 0.9 else "1. ⚠️ PILOT DEPLOYMENT - Test with subset of documents" if summary_stats['overall_accuracy'] >= 0.8 else "1. 🔧 OPTIMIZATION REQUIRED - Improve model before deployment"}
2. 📋 Establish monitoring dashboards for accuracy tracking
3. 🎯 Focus improvement efforts on challenging fields: {', '.join([f[0] for f in challenging_fields[:3]])}

### Strategic Initiatives  
- 🔄 Implement continuous evaluation pipeline
- 📊 Expand ground truth dataset for challenging document types
- ⚡ Optimize inference pipeline for production scale

---
📊 Llama-3.2-11B-Vision achieved {summary_stats['overall_accuracy']:.1%} average accuracy
"""
    
    # Save executive summary
    report_filename = f"llama_comprehensive_evaluation_report_{timestamp}.md"
    report_path = output_dir_path / report_filename
    with report_path.open('w', encoding='utf-8') as f:
        f.write(executive_summary)
    
    # Deployment Checklist
    deployment_checklist = f"""# Llama Vision Deployment Readiness Checklist

## Model Information
- **Model:** Llama-3.2-11B-Vision-Instruct
- **Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Overall Accuracy:** {summary_stats['overall_accuracy']:.1%}

## Production Readiness Checklist

### Performance Metrics
- [{'x' if summary_stats['overall_accuracy'] >= 0.8 else ' '}] Overall accuracy ≥80% ({summary_stats['overall_accuracy']:.1%})
- [{'x' if len(excellent_fields) >= 15 else ' '}] At least 15 fields with ≥90% accuracy ({len(excellent_fields)}/25)
- [{'x' if summary_stats['perfect_documents'] >= summary_stats['total_images'] * 0.3 else ' '}] At least 30% perfect documents ({summary_stats['perfect_documents']}/{summary_stats['total_images']})

### Quality Assessment
- Best Case: {summary_stats['best_performance_accuracy']:.1%} accuracy
- Worst Case: {summary_stats['worst_performance_accuracy']:.1%} accuracy

### Field Performance
- Track accuracy for critical fields: {', '.join(excellent_fields[:5])}
- Monitor challenging fields: {', '.join([f[0] for f in challenging_fields[:3]])}

## Deployment Strategy

{"✅ **APPROVED FOR PRODUCTION DEPLOYMENT**" if summary_stats['overall_accuracy'] >= 0.8 else "⚠️ **PILOT DEPLOYMENT RECOMMENDED**" if summary_stats['overall_accuracy'] >= 0.7 else "🔧 **OPTIMIZATION REQUIRED BEFORE DEPLOYMENT**"}

### Next Steps
1. {'✅ Deploy to production environment' if summary_stats['overall_accuracy'] >= 0.8 else '🧪 Run pilot with subset of documents' if summary_stats['overall_accuracy'] >= 0.7 else '🔧 Optimize model performance'}
2. 📊 Implement real-time accuracy monitoring
3. 🔄 Establish continuous evaluation pipeline
4. 📋 Create operational runbooks and troubleshooting guides

---
*Generated by Llama Vision Evaluation Pipeline*
"""
    
    # Save deployment checklist  
    checklist_filename = f"llama_deployment_checklist_{timestamp}.md"
    checklist_path = output_dir_path / checklist_filename
    with checklist_path.open('w', encoding='utf-8') as f:
        f.write(deployment_checklist)
    
    return {
        'executive_summary': report_path,
        'deployment_checklist': checklist_path
    }

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for Llama Vision key-value extraction and evaluation
    """
    print("\n" + "="*80)
    print("🦙 LLAMA VISION COMPREHENSIVE EVALUATION PIPELINE")
    print("="*80)
    
    # Ensure output directory exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Discover images
    print(f"\n📁 Discovering images in: {data_dir}")
    image_files = discover_images(data_dir)
    image_files = [f for f in image_files if 'synthetic_invoice' in Path(f).name]  # Filter for test images
    
    print(f"📷 Found {len(image_files)} images for processing")
    if not image_files:
        print("❌ No images found for processing")
        return
    
    # Process batch
    print("\n🚀 Starting batch processing...")
    results, batch_stats = process_image_batch(image_files)
    
    # Create DataFrames
    print("\n📊 Creating extraction DataFrames...")
    main_df, metadata_df = create_extraction_dataframe(results)
    
    # Save extraction results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    extraction_csv = output_dir_path / f"llama_batch_extraction_{timestamp}.csv"
    main_df.to_csv(extraction_csv, index=False)
    print(f"💾 Extraction results saved: {extraction_csv}")
    
    # Load ground truth
    print(f"\n📊 Loading ground truth from: {ground_truth_path}")
    ground_truth_data = load_ground_truth(ground_truth_path)
    
    if not ground_truth_data:
        print("❌ No ground truth data available - skipping evaluation")
        return
    
    # Perform evaluation
    print("\n🎯 Performing comprehensive evaluation...")
    evaluation_summary = evaluate_extraction_results(results, ground_truth_data)
    
    if 'error' in evaluation_summary:
        print(f"❌ Evaluation error: {evaluation_summary['error']}")
        return
    
    # Save detailed evaluation results
    eval_csv = output_dir_path / f"llama_ground_truth_evaluation_{timestamp}.csv"
    eval_df = pd.DataFrame(evaluation_summary['evaluation_data'])
    eval_df.to_csv(eval_csv, index=False)
    print(f"💾 Detailed evaluation saved: {eval_csv}")
    
    # Generate comprehensive reports
    print("\n📋 Generating comprehensive reports...")
    report_paths = generate_comprehensive_reports(evaluation_summary, output_dir_path)
    
    # Save evaluation summary as JSON
    summary_json = output_dir_path / f"llama_evaluation_summary_{timestamp}.json"
    with summary_json.open('w', encoding='utf-8') as f:
        # Make summary JSON serializable
        json_summary = {
            'overall_accuracy': evaluation_summary['overall_accuracy'],
            'total_images': evaluation_summary['total_images'],
            'perfect_documents': evaluation_summary['perfect_documents'],
            'field_accuracies': evaluation_summary['field_accuracies'],
            'best_performing_image': evaluation_summary['best_performing_image'],
            'worst_performing_image': evaluation_summary['worst_performing_image'],
            'evaluation_timestamp': timestamp
        }
        json.dump(json_summary, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 LLAMA VISION EVALUATION COMPLETE")
    print("="*80)
    print(f"🎯 Overall Accuracy: {evaluation_summary['overall_accuracy']:.1%}")
    print(f"📷 Images Processed: {evaluation_summary['total_images']}")
    print(f"🏆 Perfect Documents: {evaluation_summary['perfect_documents']}")
    print(f"📁 Results Directory: {output_dir_path}")
    print("\n📄 Generated Files:")
    print(f"   • {extraction_csv.name} - Extraction results")
    print(f"   • {eval_csv.name} - Detailed evaluation")
    print(f"   • {report_paths['executive_summary'].name} - Executive summary")
    print(f"   • {report_paths['deployment_checklist'].name} - Deployment checklist")
    print(f"   • {summary_json.name} - JSON summary")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()