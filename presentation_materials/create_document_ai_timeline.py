#!/usr/bin/env python3
"""
Generate a visual timeline for Document AI evolution
"""


import matplotlib.patches as patches
import matplotlib.pyplot as plt


def create_document_ai_timeline():
    """Create a visual timeline showing the evolution of Document AI"""
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Timeline data
    timeline_data = [
        {
            'period': 'Pre-2018',
            'title': 'OCR + Rule-based parsing',
            'start_year': 2010,
            'end_year': 2018,
            'color': '#FF6B6B',
            'description': 'Traditional OCR with\nrule-based extraction'
        },
        {
            'period': '2018-2020',
            'title': 'CNN-based document analysis',
            'start_year': 2018,
            'end_year': 2020,
            'color': '#4ECDC4',
            'description': 'Deep learning for\ndocument understanding'
        },
        {
            'period': '2020',
            'title': 'LayoutLM - First transformer for documents',
            'start_year': 2020,
            'end_year': 2020.8,
            'color': '#45B7D1',
            'description': 'Breakthrough: Text +\nLayout understanding'
        },
        {
            'period': '2021-2023',
            'title': 'LayoutLMv2, LayoutLMv3 iterations',
            'start_year': 2021,
            'end_year': 2023,
            'color': '#96CEB4',
            'description': 'Enhanced multimodal\ndocument processing'
        },
        {
            'period': '2023+',
            'title': 'Vision-Language Models (InternVL, Llama-Vision)',
            'start_year': 2023,
            'end_year': 2025,
            'color': '#FFEAA7',
            'description': 'Universal vision models\nfor any document type'
        }
    ]
    
    # Set up the timeline
    y_center = 5
    timeline_height = 1.5
    
    # Draw main timeline arrow
    ax.arrow(2009, y_center, 17, 0, head_width=0.3, head_length=0.3, 
             fc='black', ec='black', linewidth=2)
    
    # Add timeline blocks
    for _i, item in enumerate(timeline_data):
        # Calculate position
        x_start = item['start_year']
        width = item['end_year'] - item['start_year']
        
        # Draw timeline block
        rect = patches.Rectangle((x_start, y_center - timeline_height/2), 
                               width, timeline_height,
                               linewidth=2, edgecolor='black', 
                               facecolor=item['color'], alpha=0.8)
        ax.add_patch(rect)
        
        # Add period label on the block
        ax.text(x_start + width/2, y_center, item['period'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add title above the block
        ax.text(x_start + width/2, y_center + 2, item['title'], 
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               wrap=True)
        
        # Add description below the block
        ax.text(x_start + width/2, y_center - 2, item['description'], 
               ha='center', va='top', fontsize=9, style='italic')
    
    # Add year markers
    years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026]
    for year in years:
        ax.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
        ax.text(year, y_center - 3.5, str(year), ha='center', va='top', 
               fontsize=9, rotation=45)
    
    # Styling
    ax.set_xlim(2009, 2026)
    ax.set_ylim(0, 10)
    ax.set_title('Industry-Wide Evolution of Document AI', 
                fontsize=20, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add subtitle
    ax.text(2017.5, 9, '(Not ATO-specific)', ha='center', va='center', 
           fontsize=14, style='italic', color='gray')
    
    # Add current era highlight
    current_rect = patches.Rectangle((2023, y_center - timeline_height/2 - 0.2), 
                                   2, timeline_height + 0.4,
                                   linewidth=3, edgecolor='red', 
                                   facecolor='none', linestyle='--')
    ax.add_patch(current_rect)
    ax.text(2024, y_center + 3.5, 'Current Era', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = '/Users/tod/Desktop/vision_comparison/presentation_materials/presentation_diagrams/document_ai_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Also save as SVG for scalability
    svg_path = '/Users/tod/Desktop/vision_comparison/presentation_materials/presentation_diagrams/document_ai_timeline.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    
    print(f"✅ Timeline saved to: {output_path}")
    print(f"✅ SVG version saved to: {svg_path}")
    
    plt.show()

if __name__ == "__main__":
    create_document_ai_timeline()