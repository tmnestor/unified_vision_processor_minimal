# Exported Mermaid Diagrams

This directory contains rendered versions of all mermaid diagrams from the presentation, exported as both PNG and SVG formats.

## Available Diagrams

### 1. Vision Transformer Architecture
- **Files**: `Vision_Transformer_Architecture.png`, `Vision_Transformer_Architecture.svg`
- **Description**: Shows the complete ViT architecture from image patches to output features
- **Usage**: Core technical slide explaining how Vision Transformers work

### 2. LayoutLM vs Vision Transformer Comparison
- **Files**: `LayoutLM_vs_Vision_Transformer_Comparison.png`, `LayoutLM_vs_Vision_Transformer_Comparison.svg`
- **Description**: Side-by-side comparison showing LayoutLM's multi-stage pipeline vs ViT's end-to-end approach
- **Usage**: Demonstrates the architectural advantages of Vision Transformers

### 3. Self-Attention Mechanism
- **Files**: `Self_Attention_Mechanism.png`, `Self_Attention_Mechanism.svg`
- **Description**: Visualizes how self-attention links different parts of a document
- **Usage**: Explains why ViTs excel at understanding document structure

### 4. Document Processing Pipeline Comparison
- **Files**: `Document_Processing_Pipeline_Comparison.png`, `Document_Processing_Pipeline_Comparison.svg`
- **Description**: Traditional OCR pipeline vs Vision Transformer pipeline
- **Usage**: Shows the simplification and reliability benefits

### 5. Vision Transformer Components Detail
- **Files**: `Vision_Transformer_Components_Detail.png`, `Vision_Transformer_Components_Detail.svg`
- **Description**: Detailed breakdown of patch embedding and transformer blocks
- **Usage**: Technical deep-dive for interested audiences

### 6. Performance Comparison
- **Files**: `Performance_Comparison.png`, `Performance_Comparison.svg`
- **Description**: Metrics comparison chart showing LayoutLM vs ViT performance
- **Usage**: Results slide showing quantitative improvements

## File Format Guide

### PNG Files
- **Best for**: PowerPoint, Keynote, Google Slides, printed materials
- **Quality**: High resolution (1200x800px)
- **Background**: White (presentation-ready)
- **File size**: Larger but universally compatible

### SVG Files  
- **Best for**: Web presentations, HTML slides, documentation websites
- **Quality**: Vector format (infinitely scalable)
- **Background**: White
- **File size**: Smaller, perfect for web use

## Usage in Presentations

### Markdown Presentations
```markdown
![Vision Transformer Architecture](presentation_diagrams/mermaid_exports/Vision_Transformer_Architecture.png)
```

### PowerPoint/Keynote
1. Insert → Pictures → From File
2. Select the `.png` version for best compatibility
3. Resize as needed (PNG will maintain quality)

### Web/HTML Slides
```html
<img src="presentation_diagrams/mermaid_exports/Vision_Transformer_Architecture.svg" 
     alt="Vision Transformer Architecture" 
     style="max-width: 100%; height: auto;">
```

## Regenerating Diagrams

If you need to update or regenerate these diagrams:

```bash
# Run the export script
python3 export_mermaid_diagrams.py

# Or use the shell script
./export_mermaid_diagrams.sh
```

The scripts automatically:
1. Extract mermaid diagrams from `mermaid_diagrams.md`
2. Export each to both PNG and SVG formats
3. Save them in this directory with descriptive names

## Notes

- All diagrams use a white background for presentation compatibility
- PNG files are optimized for readability in presentations
- SVG files preserve all vector details and are web-optimized
- One diagram export failed (diagram_7) - this may need manual review of the mermaid syntax