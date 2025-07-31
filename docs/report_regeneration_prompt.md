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
5. Maintain the current report structure with these sections:
   - Executive Summary with correct winner analysis
   - Performance Comparison Dashboard 
   - Field-wise Extraction Analysis
   - Resource Utilization Analysis
   - Detailed Performance Metrics
   - Composite Overview
   - Recommendations & Deployment Guide
   - Technical Specifications

Ensure accuracy metrics show InternVL3 as the winner for data quality and Llama-3.2-Vision as the winner for processing speed. Include the methodology explanation that clarifies field accuracy measures useful data extraction vs "N/A" responses.

Replace the existing Vision_Model_Comparison_Report.md file and commit the changes with an appropriate git message.
```

## Important Notes

- **field_value_rates**: The meaningful metric showing percentage of documents where each field contains actual data (not "N/A")
- **field_extraction_rates**: Always 100% - both models extract all 25 fields but this doesn't indicate data quality
- **Processing Speed**: Always calculate per-image times for meaningful comparison
- **Winner Analysis**: InternVL3 for accuracy (61.8% vs 59.0%), Llama-3.2-Vision for speed (35% faster)
- **Image References**: All visualization files should use `remote_results/` path prefix

## Usage

Copy the prompt above and paste it when requesting report regeneration. This ensures consistent, accurate reporting based on the correct interpretation of the model comparison data.