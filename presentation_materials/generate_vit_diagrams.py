#!/usr/bin/env python3
"""Generate Vision Transformer diagrams for presentation."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Create output directory relative to script location
output_dir = Path(__file__).parent / "presentation_diagrams"
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def create_vit_architecture_diagram():
    """Create Vision Transformer architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Colors
    patch_color = '#E8F4FD'
    embed_color = '#B8E0D2'
    transformer_color = '#D6EADF'
    output_color = '#EAC4D5'
    
    # Draw input image
    img_rect = patches.Rectangle((0.5, 6), 1.5, 1.5, 
                                 linewidth=2, edgecolor='black', 
                                 facecolor='lightgray')
    ax.add_patch(img_rect)
    ax.text(1.25, 6.75, 'Input\nImage', ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw patches
    patch_size = 0.35
    start_x = 3
    for i in range(4):
        for j in range(4):
            x = start_x + j * (patch_size + 0.1)
            y = 5.5 + i * (patch_size + 0.1)
            patch_rect = patches.Rectangle((x, y), patch_size, patch_size,
                                         linewidth=1, edgecolor='darkblue',
                                         facecolor=patch_color)
            ax.add_patch(patch_rect)
    
    ax.text(start_x + 0.8, 7.8, 'Divide into\nPatches', ha='center', fontsize=10, weight='bold')
    
    # Draw flattening arrow
    ax.arrow(5.2, 6.75, 0.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Draw patch embeddings
    embed_start_x = 6.5
    for i in range(9):  # Show 9 embeddings + ...
        y = 5.2 + i * 0.3
        if i < 8:
            embed_rect = patches.Rectangle((embed_start_x, y), 1.2, 0.25,
                                         linewidth=1, edgecolor='darkgreen',
                                         facecolor=embed_color)
            ax.add_patch(embed_rect)
        else:
            ax.text(embed_start_x + 0.6, y + 0.125, '...', ha='center', va='center', fontsize=12)
    
    ax.text(embed_start_x + 0.6, 8.2, 'Linear\nProjection\n+ Position', ha='center', fontsize=10, weight='bold')
    
    # Draw transformer blocks
    ax.arrow(7.9, 6.75, 0.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    trans_start_x = 9
    block_height = 1.5
    for i in range(3):
        y = 5 + i * (block_height + 0.3)
        if i < 2:
            # Transformer block
            trans_rect = patches.Rectangle((trans_start_x, y), 2, block_height,
                                         linewidth=2, edgecolor='darkgreen',
                                         facecolor=transformer_color)
            ax.add_patch(trans_rect)
            ax.text(trans_start_x + 1, y + 0.75, f'Transformer\nBlock {i+1}', 
                    ha='center', va='center', fontsize=9, weight='bold')
        else:
            ax.text(trans_start_x + 1, y + 0.75, '...', ha='center', va='center', fontsize=16)
    
    # Draw output
    ax.arrow(11.2, 6.75, 0.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    output_rect = patches.Rectangle((12.5, 6), 1.5, 1.5,
                                  linewidth=2, edgecolor='purple',
                                  facecolor=output_color)
    ax.add_patch(output_rect)
    ax.text(13.25, 6.75, 'Output\nFeatures', ha='center', va='center', fontsize=10, weight='bold')
    
    # Add title and labels
    ax.text(7, 9.5, 'Vision Transformer Architecture', fontsize=18, weight='bold', ha='center')
    ax.text(7, 4, 'Image → Patches → Embeddings → Transformer → Output', 
            fontsize=12, ha='center', style='italic')
    
    # Clean up
    ax.set_xlim(0, 14)
    ax.set_ylim(3.5, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vit_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_attention_mechanism_diagram():
    """Create self-attention mechanism visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors
    query_color = '#FFB6C1'
    key_color = '#98FB98'
    value_color = '#87CEEB'
    attention_color = '#FFE4B5'
    
    # Draw patches
    patches_y = 6
    patch_positions = []
    for i in range(6):
        x = 1 + i * 1.5
        rect = patches.Rectangle((x, patches_y), 1, 1,
                               linewidth=2, edgecolor='black',
                               facecolor='lightgray')
        ax.add_patch(rect)
        ax.text(x + 0.5, patches_y + 0.5, f'P{i+1}', ha='center', va='center', fontsize=10)
        patch_positions.append((x + 0.5, patches_y))
    
    # Highlight query patch
    query_patch = patches.Rectangle((2.5, patches_y), 1, 1,
                                  linewidth=3, edgecolor='red',
                                  facecolor=query_color, alpha=0.7)
    ax.add_patch(query_patch)
    ax.text(3, patches_y - 0.5, 'Query', ha='center', fontsize=10, weight='bold', color='red')
    
    # Draw attention connections
    query_x = 3
    for i, (px, py) in enumerate(patch_positions):
        if i != 1:  # Skip self
            # Calculate attention weight (simulated)
            weight = 0.1 + 0.8 * np.exp(-abs(i - 1) / 2)
            ax.plot([query_x, px], [patches_y + 1, py + 1],
                   'k-', alpha=weight, linewidth=2 * weight)
            ax.text((query_x + px) / 2, (patches_y + 1 + py + 1) / 2 + 0.2,
                   f'{weight:.2f}', ha='center', fontsize=8)
    
    # Add labels
    ax.text(5, 8.5, 'Self-Attention: Each Patch Attends to All Others', 
            fontsize=16, weight='bold', ha='center')
    ax.text(5, 4.5, 'Attention weights show relevance between patches', 
            fontsize=12, ha='center', style='italic')
    
    # Add formula
    ax.text(5, 3, r'Attention(Q, K, V) = softmax($\frac{QK^T}{\sqrt{d_k}}$)V',
            fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor=attention_color))
    
    # Clean up
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_vit_vs_cnn_comparison():
    """Create ViT vs CNN comparison diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # CNN side
    ax1.text(0.5, 0.95, 'CNN Approach', transform=ax1.transAxes, 
             fontsize=16, weight='bold', ha='center')
    
    # Draw CNN layers
    layer_colors = ['#FFE4E1', '#FFD700', '#98FB98', '#87CEEB']
    layer_names = ['Conv + Pool', 'Conv + Pool', 'Conv + Pool', 'Fully Connected']
    layer_sizes = [(3, 2), (2.5, 1.5), (2, 1), (1.5, 0.5)]
    
    y_pos = 0.7
    for i, (name, size, color) in enumerate(zip(layer_names, layer_sizes, layer_colors, strict=False)):
        rect = patches.Rectangle((0.5 - size[0]/2, y_pos - size[1]/2), size[0], size[1],
                               transform=ax1.transAxes, linewidth=2, 
                               edgecolor='black', facecolor=color)
        ax1.add_patch(rect)
        ax1.text(0.5, y_pos, name, transform=ax1.transAxes,
                ha='center', va='center', fontsize=10, weight='bold')
        y_pos -= 0.2
        
        if i < len(layer_names) - 1:
            ax1.arrow(0.5, y_pos + 0.08, 0, -0.06, transform=ax1.transAxes,
                     head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    ax1.text(0.5, 0.05, 'Local Features → Hierarchical Processing', 
             transform=ax1.transAxes, ha='center', fontsize=10, style='italic')
    
    # ViT side
    ax2.text(0.5, 0.95, 'ViT Approach', transform=ax2.transAxes, 
             fontsize=16, weight='bold', ha='center')
    
    # Draw ViT components
    # Patches
    patch_y = 0.75
    for i in range(4):
        for j in range(4):
            x = 0.3 + j * 0.1
            y = patch_y - i * 0.08
            rect = patches.Rectangle((x, y), 0.08, 0.06,
                                   transform=ax2.transAxes, linewidth=1,
                                   edgecolor='darkblue', facecolor='#E8F4FD')
            ax2.add_patch(rect)
    
    ax2.text(0.5, 0.85, 'Image Patches', transform=ax2.transAxes,
            ha='center', fontsize=10)
    
    # Arrow
    ax2.arrow(0.5, 0.65, 0, -0.05, transform=ax2.transAxes,
             head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    # Transformer
    trans_rect = patches.Rectangle((0.2, 0.35), 0.6, 0.25,
                                 transform=ax2.transAxes, linewidth=2,
                                 edgecolor='darkgreen', facecolor='#D6EADF')
    ax2.add_patch(trans_rect)
    ax2.text(0.5, 0.475, 'Transformer\n(Global Attention)', transform=ax2.transAxes,
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrow
    ax2.arrow(0.5, 0.33, 0, -0.05, transform=ax2.transAxes,
             head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    # Output
    output_rect = patches.Rectangle((0.35, 0.15), 0.3, 0.12,
                                  transform=ax2.transAxes, linewidth=2,
                                  edgecolor='purple', facecolor='#EAC4D5')
    ax2.add_patch(output_rect)
    ax2.text(0.5, 0.21, 'Output', transform=ax2.transAxes,
            ha='center', va='center', fontsize=10, weight='bold')
    
    ax2.text(0.5, 0.05, 'Global Context → Direct Processing', 
             transform=ax2.transAxes, ha='center', fontsize=10, style='italic')
    
    # Clean up
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vit_vs_cnn.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_document_processing_comparison():
    """Create document processing pipeline comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Traditional OCR Pipeline
    ax1.text(0.5, 0.9, 'Traditional OCR + Parsing Pipeline', 
             transform=ax1.transAxes, fontsize=14, weight='bold', ha='center')
    
    # Pipeline steps
    steps = ['Document\nImage', 'OCR\nEngine', 'Raw\nText', 'Regex/\nRules', 'Parsed\nFields']
    colors = ['#FFE4E1', '#FFD700', '#98FB98', '#87CEEB', '#DDA0DD']
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for i, (step, x, color) in enumerate(zip(steps, x_positions, colors, strict=False)):
        rect = patches.FancyBboxPatch((x - 0.08, 0.35), 0.16, 0.3,
                                    boxstyle="round,pad=0.02",
                                    transform=ax1.transAxes,
                                    linewidth=2, edgecolor='black',
                                    facecolor=color)
        ax1.add_patch(rect)
        ax1.text(x, 0.5, step, transform=ax1.transAxes,
                ha='center', va='center', fontsize=10, weight='bold')
        
        if i < len(steps) - 1:
            ax1.arrow(x + 0.09, 0.5, 0.11, 0, transform=ax1.transAxes,
                     head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    # Add failure points
    ax1.text(0.2, 0.2, '❌ OCR errors', transform=ax1.transAxes,
            ha='center', fontsize=9, color='red')
    ax1.text(0.4, 0.2, '❌ Layout loss', transform=ax1.transAxes,
            ha='center', fontsize=9, color='red')
    ax1.text(0.6, 0.2, '❌ Rule brittleness', transform=ax1.transAxes,
            ha='center', fontsize=9, color='red')
    
    # Vision Transformer Pipeline
    ax2.text(0.5, 0.9, 'Vision Transformer Pipeline', 
             transform=ax2.transAxes, fontsize=14, weight='bold', ha='center')
    
    # Simpler pipeline
    vit_steps = ['Document\nImage', 'Vision\nTransformer\n+ LM', 'Structured\nFields']
    vit_colors = ['#FFE4E1', '#90EE90', '#DDA0DD']
    vit_x = [0.2, 0.5, 0.8]
    
    for i, (step, x, color) in enumerate(zip(vit_steps, vit_x, vit_colors, strict=False)):
        if i == 1:  # Make middle box larger
            rect = patches.FancyBboxPatch((x - 0.12, 0.3), 0.24, 0.4,
                                        boxstyle="round,pad=0.02",
                                        transform=ax2.transAxes,
                                        linewidth=3, edgecolor='darkgreen',
                                        facecolor=color)
        else:
            rect = patches.FancyBboxPatch((x - 0.08, 0.35), 0.16, 0.3,
                                        boxstyle="round,pad=0.02",
                                        transform=ax2.transAxes,
                                        linewidth=2, edgecolor='black',
                                        facecolor=color)
        ax2.add_patch(rect)
        ax2.text(x, 0.5, step, transform=ax2.transAxes,
                ha='center', va='center', fontsize=10, weight='bold')
        
        if i < len(vit_steps) - 1:
            ax2.arrow(x + 0.13 if i == 1 else 0.09, 0.5, 
                     0.14 if i == 0 else 0.18, 0, 
                     transform=ax2.transAxes,
                     head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    # Add benefits
    ax2.text(0.5, 0.15, '✅ End-to-end learning  ✅ Layout preserved  ✅ Self-adapting', 
             transform=ax2.transAxes, ha='center', fontsize=10, color='green', weight='bold')
    
    # Clean up
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'document_processing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_project_results_visualization():
    """Create visualization of our project results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Model comparison
    models = ['InternVL3-2B', 'Llama-3.2-11B']
    accuracy = [59.4, 59.0]
    speed = [22.6, 24.9]
    memory = [2.6, 13.3]
    
    # Accuracy comparison
    bars1 = ax1.bar(models, accuracy, color=['#90EE90', '#87CEEB'])
    ax1.set_ylabel('Field Accuracy (%)', fontsize=12)
    ax1.set_title('Extraction Accuracy', fontsize=14, weight='bold')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars1, accuracy, strict=False):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=10)
    
    # Speed comparison
    bars2 = ax2.bar(models, speed, color=['#90EE90', '#87CEEB'])
    ax2.set_ylabel('Processing Time (s/image)', fontsize=12)
    ax2.set_title('Processing Speed', fontsize=14, weight='bold')
    ax2.set_ylim(0, 30)
    for bar, val in zip(bars2, speed, strict=False):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}s', ha='center', fontsize=10)
    
    # Memory usage
    bars3 = ax3.bar(models, memory, color=['#90EE90', '#87CEEB'])
    ax3.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax3.set_title('Memory Efficiency', fontsize=14, weight='bold')
    ax3.set_ylim(0, 16)
    ax3.axhline(y=16, color='red', linestyle='--', label='V100 Limit')
    for bar, val in zip(bars3, memory, strict=False):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val}GB', ha='center', fontsize=10)
    ax3.legend()
    
    # Field extraction breakdown
    field_categories = ['Core\nFields', 'Financial\nFields', 'Business\nInfo', 'Banking\nFields']
    internvl_scores = [95, 88, 92, 65]
    llama_scores = [93, 85, 90, 62]
    
    x = np.arange(len(field_categories))
    width = 0.35
    
    bars4_1 = ax4.bar(x - width/2, internvl_scores, width, label='InternVL3', color='#90EE90')
    bars4_2 = ax4.bar(x + width/2, llama_scores, width, label='Llama-3.2', color='#87CEEB')
    
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Field Category Performance', fontsize=14, weight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(field_categories)
    ax4.legend()
    ax4.set_ylim(0, 100)
    
    plt.suptitle('Vision Transformer Model Performance Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'project_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_layoutlm_vs_vit_architecture():
    """Create LayoutLM vs ViT architecture comparison diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # LayoutLM Architecture (Left)
    ax1.text(0.5, 0.95, 'LayoutLM Architecture', transform=ax1.transAxes, 
             fontsize=16, weight='bold', ha='center')
    
    # Document input
    doc_rect = patches.Rectangle((0.2, 0.8), 0.6, 0.1, 
                                linewidth=2, edgecolor='black', 
                                facecolor='lightgray', transform=ax1.transAxes)
    ax1.add_patch(doc_rect)
    ax1.text(0.5, 0.85, 'Document Image', transform=ax1.transAxes,
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Three branches
    ax1.arrow(0.3, 0.78, 0, -0.08, transform=ax1.transAxes,
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax1.arrow(0.5, 0.78, 0, -0.08, transform=ax1.transAxes,
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax1.arrow(0.7, 0.78, 0, -0.08, transform=ax1.transAxes,
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # OCR box
    ocr_rect = patches.Rectangle((0.1, 0.55), 0.25, 0.12,
                               linewidth=2, edgecolor='red',
                               facecolor='#FFE4E1', transform=ax1.transAxes)
    ax1.add_patch(ocr_rect)
    ax1.text(0.225, 0.61, 'OCR\nEngine', transform=ax1.transAxes,
            ha='center', va='center', fontsize=9, weight='bold')
    
    # CNN box
    cnn_rect = patches.Rectangle((0.375, 0.55), 0.25, 0.12,
                               linewidth=2, edgecolor='blue',
                               facecolor='#E0E0FF', transform=ax1.transAxes)
    ax1.add_patch(cnn_rect)
    ax1.text(0.5, 0.61, 'CNN\nFeatures', transform=ax1.transAxes,
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Coordinate box
    coord_rect = patches.Rectangle((0.65, 0.55), 0.25, 0.12,
                                 linewidth=2, edgecolor='green',
                                 facecolor='#E0FFE0', transform=ax1.transAxes)
    ax1.add_patch(coord_rect)
    ax1.text(0.775, 0.61, 'Layout\nCoords', transform=ax1.transAxes,
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows to LayoutLM
    ax1.arrow(0.225, 0.53, 0.075, -0.08, transform=ax1.transAxes,
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax1.arrow(0.5, 0.53, 0, -0.08, transform=ax1.transAxes,
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax1.arrow(0.775, 0.53, -0.075, -0.08, transform=ax1.transAxes,
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # LayoutLM box
    layout_rect = patches.Rectangle((0.2, 0.3), 0.6, 0.15,
                                  linewidth=3, edgecolor='purple',
                                  facecolor='#FFE4FF', transform=ax1.transAxes)
    ax1.add_patch(layout_rect)
    ax1.text(0.5, 0.375, 'LayoutLM\nTransformer', transform=ax1.transAxes,
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Output
    ax1.arrow(0.5, 0.28, 0, -0.08, transform=ax1.transAxes,
             head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    output_rect = patches.Rectangle((0.35, 0.1), 0.3, 0.1,
                                  linewidth=2, edgecolor='black',
                                  facecolor='#FFFACD', transform=ax1.transAxes)
    ax1.add_patch(output_rect)
    ax1.text(0.5, 0.15, 'Extracted Fields', transform=ax1.transAxes,
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Add failure points
    ax1.text(0.225, 0.5, '❌', transform=ax1.transAxes,
            ha='center', fontsize=16, color='red')
    ax1.text(0.5, 0.25, '⚠️', transform=ax1.transAxes,
            ha='center', fontsize=16, color='orange')
    
    ax1.text(0.5, 0.02, 'Multiple Components, Multiple Failure Points', 
             transform=ax1.transAxes, ha='center', fontsize=10, 
             style='italic', color='red')
    
    # Vision Transformer Architecture (Right)
    ax2.text(0.5, 0.95, 'Vision Transformer Architecture', transform=ax2.transAxes, 
             fontsize=16, weight='bold', ha='center')
    
    # Document input
    doc_rect2 = patches.Rectangle((0.2, 0.8), 0.6, 0.1, 
                                 linewidth=2, edgecolor='black', 
                                 facecolor='lightgray', transform=ax2.transAxes)
    ax2.add_patch(doc_rect2)
    ax2.text(0.5, 0.85, 'Document Image', transform=ax2.transAxes,
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Direct arrow
    ax2.arrow(0.5, 0.78, 0, -0.18, transform=ax2.transAxes,
             head_width=0.03, head_length=0.03, fc='green', ec='green', linewidth=3)
    
    # Vision Transformer box
    vit_rect = patches.Rectangle((0.15, 0.35), 0.7, 0.25,
                               linewidth=3, edgecolor='darkgreen',
                               facecolor='#90EE90', transform=ax2.transAxes)
    ax2.add_patch(vit_rect)
    ax2.text(0.5, 0.475, 'Vision Transformer\n+\nLanguage Model', transform=ax2.transAxes,
            ha='center', va='center', fontsize=14, weight='bold')
    
    # Output
    ax2.arrow(0.5, 0.33, 0, -0.13, transform=ax2.transAxes,
             head_width=0.03, head_length=0.03, fc='green', ec='green', linewidth=3)
    
    output_rect2 = patches.Rectangle((0.35, 0.1), 0.3, 0.1,
                                   linewidth=2, edgecolor='black',
                                   facecolor='#FFFACD', transform=ax2.transAxes)
    ax2.add_patch(output_rect2)
    ax2.text(0.5, 0.15, 'Extracted Fields', transform=ax2.transAxes,
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Add success indicator
    ax2.text(0.85, 0.475, '✅', transform=ax2.transAxes,
            ha='center', fontsize=20, color='green')
    
    ax2.text(0.5, 0.02, 'Single Model, End-to-End Processing', 
             transform=ax2.transAxes, ha='center', fontsize=10, 
             style='italic', color='green', weight='bold')
    
    # Clean up
    ax1.axis('off')
    ax2.axis('off')
    
    plt.suptitle('LayoutLM vs Vision Transformer: Architectural Comparison', 
                 fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'layoutlm_vs_vit_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all diagrams."""
    print("Generating Vision Transformer presentation diagrams...")
    
    create_vit_architecture_diagram()
    print("✓ Created ViT architecture diagram")
    
    create_attention_mechanism_diagram()
    print("✓ Created attention mechanism diagram")
    
    create_vit_vs_cnn_comparison()
    print("✓ Created ViT vs CNN comparison")
    
    create_document_processing_comparison()
    print("✓ Created document processing comparison")
    
    create_project_results_visualization()
    print("✓ Created project results visualization")
    
    create_layoutlm_vs_vit_architecture()
    print("✓ Created LayoutLM vs ViT architecture comparison")
    
    print(f"\nAll diagrams saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()