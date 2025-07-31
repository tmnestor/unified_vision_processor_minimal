# Vision Model Comparison Report Regeneration Prompt

This document contains the standardized prompt for regenerating the Vision Model Comparison Report with correct metrics and analysis.

## Prompt for Claude

```
Please regenerate the Vision_Model_Comparison_Report.md file using the latest data from remote_results/comparison_results_full.json. 

Key requirements:
1. Use field_value_rates as the primary accuracy metric (meaningful data vs "N/A" responses)
2. Ignore field_extraction_rates (always 100%, provides no useful differentiation)
3. Calculate per-image processing times by dividing total execution time by number of images
4. Reference all visualization images with remote_results/ prefix
5. **CRITICAL: Use specific model names and correct specifications throughout the report**:
   - "Llama-3.2-11B-Vision-Instruct" (11B parameters, by Meta)
   - "InternVL3-2B" (2B parameters, by OpenGVLab) - NOT 8B parameters
   - Include these full model names prominently in the title and executive summary
6. Fix dashboard visualization heading corruption if present
7. Maintain the current report structure with these sections:
   - Executive Summary with correct winner analysis
   - Performance Comparison Dashboard 
   - Field-wise Extraction Analysis
   - Resource Utilization Analysis (including production memory requirements)
   - Production POD Sizing Requirements (NEW - critical for deployment)
   - Detailed Performance Metrics
   - Composite Overview
   - Recommendations & Deployment Guide
   - Technical Specifications

Ensure analysis and visualizations accurately reflect the data in remote_results/comparison_results_full.json. Include the methodology explanation that clarifies field accuracy measures useful data extraction vs "N/A" responses.

Replace the existing Vision_Model_Comparison_Report.md file and commit the changes with an appropriate git message.
```

## Important Notes

- **field_value_rates**: The meaningful metric showing percentage of documents where each field contains actual data (not "N/A")
- **field_extraction_rates**: Always 100% - both models extract all 25 fields but this doesn't indicate data quality
- **Processing Speed**: Always calculate per-image times for meaningful comparison
- **Winner Analysis**: Use current data-driven winner analysis with specific model names (Llama-3.2-11B-Vision-Instruct vs InternVL3-2B)
- **Memory Requirements**: Critical for POD sizing - include both CPU memory and GPU VRAM analysis
- **Production Deployment**: Include Kubernetes POD resource specifications and cost analysis
- **Image References**: All visualization files should use `remote_results/` path prefix

## Generated Visualizations

The system now generates these critical visualizations:
- `model_performance_dashboard.png` - Overall performance comparison
- `field_accuracy_heatmap_25fields.png` - Field-wise accuracy analysis
- `field_category_analysis.png` - Performance by field categories
- `v100_vram_usage_comparison.png` - GPU memory usage comparison
- `production_memory_requirements.png` - **NEW: Critical for POD sizing decisions**
- `composite_overview_2x2.png` - Summary dashboard

## Usage

Copy the prompt above and paste it when requesting report regeneration. This ensures consistent, accurate reporting based on the correct interpretation of the model comparison data.