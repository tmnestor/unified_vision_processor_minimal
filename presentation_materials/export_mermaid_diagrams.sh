#!/bin/bash

# Script to export Mermaid diagrams to PNG/SVG files
# This script extracts mermaid diagrams from markdown files and converts them to images

set -e

echo "🎨 Mermaid Diagram Export Script"
echo "================================"

# Check if mermaid-cli is installed
if ! command -v mmdc &> /dev/null; then
    echo "📦 Installing Mermaid CLI..."
    npm install -g @mermaid-js/mermaid-cli
fi

# Create output directory
DIAGRAM_DIR="presentation_diagrams/mermaid_exports"
mkdir -p "$DIAGRAM_DIR"

# Extract and export diagrams from mermaid_diagrams.md
echo "🔍 Processing mermaid_diagrams.md..."

# Function to extract mermaid blocks and save as separate files
extract_mermaid_blocks() {
    local input_file="$1"
    local block_num=1
    local in_mermaid_block=false
    local current_diagram=""
    local diagram_title=""
    
    while IFS= read -r line; do
        if [[ $line =~ ^##[[:space:]]*[0-9]+\.[[:space:]]*(.+)$ ]]; then
            # Extract diagram title from header
            diagram_title="${BASH_REMATCH[1]}"
            diagram_title=$(echo "$diagram_title" | sed 's/[^a-zA-Z0-9 ]//g' | sed 's/ /_/g')
        elif [[ $line == '```mermaid' ]]; then
            in_mermaid_block=true
            current_diagram=""
        elif [[ $line == '```' ]] && [[ $in_mermaid_block == true ]]; then
            in_mermaid_block=false
            
            # Save mermaid block to temporary file
            local temp_file="/tmp/mermaid_${block_num}.mmd"
            echo "$current_diagram" > "$temp_file"
            
            # Generate filename
            local base_name="${diagram_title:-diagram_${block_num}}"
            
            echo "  📊 Exporting: $base_name"
            
            # Export to PNG (high quality)
            mmdc -i "$temp_file" -o "$DIAGRAM_DIR/${base_name}.png" \
                -t default -b white --width 1200 --height 800 \
                2>/dev/null || echo "    ⚠️  PNG export failed"
            
            # Export to SVG (vector format)
            mmdc -i "$temp_file" -o "$DIAGRAM_DIR/${base_name}.svg" \
                -t default -b white \
                2>/dev/null || echo "    ⚠️  SVG export failed"
            
            # Clean up temp file
            rm -f "$temp_file"
            
            block_num=$((block_num + 1))
        elif [[ $in_mermaid_block == true ]]; then
            current_diagram="$current_diagram"$'\n'"$line"
        fi
    done < "$input_file"
}

# Process the mermaid diagrams file
if [[ -f "mermaid_diagrams.md" ]]; then
    extract_mermaid_blocks "mermaid_diagrams.md"
else
    echo "❌ mermaid_diagrams.md not found!"
    exit 1
fi

echo ""
echo "✅ Export complete!"
echo "📁 Diagrams saved to: $DIAGRAM_DIR"
echo ""
echo "📋 Generated files:"
ls -la "$DIAGRAM_DIR"

echo ""
echo "💡 Usage in presentations:"
echo "   - Use PNG files for PowerPoint/Keynote (high quality, good compatibility)"
echo "   - Use SVG files for web presentations (scalable, smaller file size)"
echo "   - Reference in markdown: ![Diagram Name](presentation_diagrams/mermaid_exports/diagram_name.png)"