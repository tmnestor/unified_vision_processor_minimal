#!/usr/bin/env python3
"""Fix Marp slides to prevent content overflow in PowerPoint."""

import re
from pathlib import Path


def fix_slide_overflow(content):
    """Fix slides that have too much content."""
    
    # Split into slides
    slides = content.split('\n---\n')
    
    fixed_slides = []
    
    for i, slide in enumerate(slides):
        lines = slide.strip().split('\n')
        
        # Skip the YAML frontmatter
        if i == 0:
            fixed_slides.append(slide)
            continue
            
        # Count content lines (excluding empty lines)
        content_lines = [line for line in lines if line.strip()]
        
        # If slide has too much content, split it
        if len(content_lines) > 15:  # Threshold for PowerPoint
            # Find the slide title
            title_line = None
            for j, line in enumerate(lines):
                if line.startswith('###'):
                    title_line = j
                    break
            
            if title_line is not None:
                # Keep essential content on main slide
                main_content = []
                overflow_content = []
                
                # Always keep the title
                main_content.extend(lines[:title_line+1])
                
                # Process remaining content
                remaining = lines[title_line+1:]
                line_count = 0
                
                for line in remaining:
                    # Move speaker notes to comments
                    if line.startswith('**Notes**:'):
                        main_content.append('')
                        main_content.append('<!-- Speaker Notes:')
                        main_content.append(line.replace('**Notes**:', ''))
                        main_content.append('-->')
                        continue
                    
                    # Keep images and tables on main slide
                    if '![' in line or line.strip().startswith('|'):
                        main_content.append(line)
                        continue
                    
                    # Move detailed lists to continuation slides
                    if line.strip().startswith('-') and line_count > 8:
                        overflow_content.append(line)
                    else:
                        main_content.append(line)
                        if line.strip():
                            line_count += 1
                
                # Create main slide
                fixed_slides.append('\n'.join(main_content))
                
                # Create continuation slide if needed
                if overflow_content:
                    cont_title = lines[title_line].replace('###', '###') + ' (continued)'
                    cont_slide = [cont_title, ''] + overflow_content
                    fixed_slides.append('\n'.join(cont_slide))
            else:
                fixed_slides.append(slide)
        else:
            fixed_slides.append(slide)
    
    return '\n---\n'.join(fixed_slides)


def simplify_tables(content):
    """Simplify large tables to fit on slides."""
    # Find tables and limit their size
    lines = content.split('\n')
    in_table = False
    table_lines = []
    new_lines = []
    
    for line in lines:
        if '|' in line and '---' not in line:
            in_table = True
            table_lines.append(line)
        elif in_table and '|' not in line:
            # End of table
            in_table = False
            # Keep only first 5 rows of table
            if len(table_lines) > 6:
                new_lines.extend(table_lines[:6])
                new_lines.append('| ... | ... | ... |')
            else:
                new_lines.extend(table_lines)
            table_lines = []
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_code_blocks(content):
    """Reduce code block size for slides."""
    # Pattern to find code blocks
    pattern = r'```(\w*)\n(.*?)\n```'
    
    def replace_code(match):
        lang = match.group(1)
        code = match.group(2)
        lines = code.split('\n')
        
        # If code is too long, truncate it
        if len(lines) > 10:
            truncated = lines[:8]
            truncated.append('// ... (code continues)')
            return f'```{lang}\n' + '\n'.join(truncated) + '\n```'
        return match.group(0)
    
    return re.sub(pattern, replace_code, content, flags=re.DOTALL)


def main():
    """Fix the presentation file."""
    file_path = Path('vision_transformers_slides.md')
    
    # Read the content
    content = file_path.read_text()
    
    # Apply fixes
    content = fix_slide_overflow(content)
    content = simplify_tables(content)
    content = fix_code_blocks(content)
    
    # Write back
    output_path = Path('vision_transformers_slides_fixed.md')
    output_path.write_text(content)
    
    print(f"✅ Fixed presentation saved to: {output_path}")
    print("📝 Changes made:")
    print("  - Split overflowing slides")
    print("  - Moved speaker notes to comments")
    print("  - Simplified large tables")
    print("  - Truncated long code blocks")
    print("\nNow run:")
    print(f"  marp {output_path} --pptx --allow-local-files -o presentation.pptx")


if __name__ == "__main__":
    main()