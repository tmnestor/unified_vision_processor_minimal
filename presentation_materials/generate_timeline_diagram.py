#!/usr/bin/env python3
"""
Document AI Timeline Generator
=============================
Creates a visual timeline showing the evolution of Document AI technologies.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def create_timeline():
    """Create a visual timeline of Document AI evolution."""
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Timeline data
    timeline_data = [
        {"year": "Pre-2018", "tech": "OCR + Rules", "color": "#ff6b6b", "desc": "Rule-based parsing"},
        {"year": "2018-2020", "tech": "CNN-based", "color": "#4ecdc4", "desc": "Document analysis"},
        {"year": "2020", "tech": "LayoutLM v1", "color": "#45b7d1", "desc": "First transformer\nfor documents"},
        {"year": "2021-2023", "tech": "LayoutLM v2/v3", "color": "#96ceb4", "desc": "Iterations &\nimprovements"},
        {"year": "2023+", "tech": "Vision-Language", "color": "#feca57", "desc": "InternVL,\nLlama-Vision"}
    ]
    
    # Set up the plot
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Title
    ax.text(5, 7.5, 'Industry-Wide Evolution of Document AI', 
            fontsize=24, fontweight='bold', ha='center', va='center')
    
    # Subtitle
    ax.text(5, 7, 'Timeline (Not ATO-specific)', 
            fontsize=16, ha='center', va='center', style='italic', color='#666666')
    
    # Draw the main timeline line
    ax.plot([1, 9], [4, 4], 'k-', linewidth=3, alpha=0.3)
    
    # Calculate positions
    x_positions = np.linspace(1.5, 8.5, len(timeline_data))
    
    # Draw timeline elements
    for i, (data, x_pos) in enumerate(zip(timeline_data, x_positions, strict=False)):
        # Draw vertical line
        ax.plot([x_pos, x_pos], [3.7, 4.3], 'k-', linewidth=2, alpha=0.7)
        
        # Draw the technology box
        if i % 2 == 0:  # Alternate above and below
            y_box = 5.5
            y_text = 5.5
            y_desc = 5.0
            y_year = 2.5
        else:
            y_box = 2.5
            y_text = 2.5
            y_desc = 3.0
            y_year = 5.5
        
        # Create fancy box for technology name
        box = FancyBboxPatch((x_pos-0.6, y_box-0.3), 1.2, 0.6,
                           boxstyle="round,pad=0.1",
                           facecolor=data['color'],
                           edgecolor='black',
                           linewidth=1.5,
                           alpha=0.9)
        ax.add_patch(box)
        
        # Add technology name
        ax.text(x_pos, y_text, data['tech'], 
                fontsize=11, fontweight='bold', ha='center', va='center',
                color='white' if data['color'] in ['#ff6b6b', '#45b7d1'] else 'black')
        
        # Add description
        ax.text(x_pos, y_desc, data['desc'], 
                fontsize=9, ha='center', va='center', color='#444444')
        
        # Add year
        year_box = FancyBboxPatch((x_pos-0.4, y_year-0.15), 0.8, 0.3,
                                boxstyle="round,pad=0.05",
                                facecolor='#f8f9fa',
                                edgecolor='#666666',
                                linewidth=1)
        ax.add_patch(year_box)
        
        ax.text(x_pos, y_year, data['year'], 
                fontsize=10, fontweight='bold', ha='center', va='center', color='#333333')
    
    # Add arrows showing progression
    for i in range(len(x_positions) - 1):
        arrow = patches.FancyArrowPatch((x_positions[i] + 0.6, 4), 
                                      (x_positions[i + 1] - 0.6, 4),
                                      arrowstyle='->', 
                                      mutation_scale=20, 
                                      alpha=0.6,
                                      color='#666666')
        ax.add_patch(arrow)
    
    # Add current focus indicator
    current_box = FancyBboxPatch((x_positions[-1]-0.8, 1.2), 1.6, 0.4,
                               boxstyle="round,pad=0.1",
                               facecolor='#ff6b6b',
                               edgecolor='black',
                               linewidth=2,
                               alpha=0.9)
    ax.add_patch(current_box)
    
    ax.text(x_positions[-1], 1.4, 'Current Focus', 
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    
    # Add note about industry trend
    ax.text(5, 0.5, 'Global industry trend: Moving from fragmented OCR-dependent pipelines\nto unified vision-language models',
            fontsize=12, ha='center', va='center', style='italic', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the timeline diagram."""
    print("🎨 Generating Document AI Evolution Timeline...")
    
    # Create the timeline
    fig = create_timeline()
    
    # Save the figure
    output_path = "presentation_diagrams/document_ai_timeline.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Timeline saved to: {output_path}")
    print("📏 Resolution: 300 DPI, optimized for presentations")
    
    # Also save as SVG for scalability
    svg_path = "presentation_diagrams/document_ai_timeline.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"✅ SVG version saved to: {svg_path}")
    
    plt.close()

if __name__ == "__main__":
    main()