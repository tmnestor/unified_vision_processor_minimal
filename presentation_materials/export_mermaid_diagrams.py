#!/usr/bin/env python3
"""
Mermaid Diagram Export Tool
==========================
Extracts mermaid diagrams from markdown files and exports them as PNG/SVG images.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def check_mermaid_cli() -> bool:
    """Check if mermaid-cli is installed."""
    try:
        subprocess.run(["mmdc", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_mermaid_cli() -> bool:
    """Install mermaid-cli using npm."""
    try:
        print("📦 Installing Mermaid CLI...")
        subprocess.run(["npm", "install", "-g", "@mermaid-js/mermaid-cli"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install mermaid-cli. Please install manually:")
        print("   npm install -g @mermaid-js/mermaid-cli")
        return False


def extract_mermaid_blocks(file_path: Path) -> List[Tuple[str, str, str]]:
    """Extract mermaid blocks from markdown file.

    Returns:
        List of tuples: (title, diagram_code, section_number)
    """
    with file_path.open("r", encoding="utf-8") as f:
        content = f.read()

    diagrams = []
    current_title = ""
    section_num = 1

    # Split content by mermaid blocks
    parts = re.split(r"```mermaid\n(.*?)\n```", content, flags=re.DOTALL)

    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            # Look for title in the preceding text
            preceding_text = parts[i]
            diagram_code = parts[i + 1].strip()

            # Extract title from the last header before this diagram
            headers = re.findall(r"^##\s*\d+\.\s*(.+)$", preceding_text, re.MULTILINE)
            if headers:
                current_title = headers[-1]
                # Clean title for filename
                clean_title = re.sub(r"[^\w\s-]", "", current_title)
                clean_title = re.sub(r"[-\s]+", "_", clean_title).strip("_")
            else:
                clean_title = f"diagram_{section_num}"

            if diagram_code:
                diagrams.append((clean_title, diagram_code, str(section_num)))
                section_num += 1

    return diagrams


def export_diagram(title: str, code: str, output_dir: Path) -> bool:
    """Export a single mermaid diagram to PNG and SVG."""
    # Create temporary mermaid file
    temp_file = output_dir / f"temp_{title}.mmd"

    try:
        # Write mermaid code to temp file
        with temp_file.open("w", encoding="utf-8") as f:
            f.write(code)

        success = True

        # Export to PNG (high quality for presentations)
        png_file = output_dir / f"{title}.png"
        try:
            subprocess.run(
                [
                    "mmdc",
                    "-i",
                    str(temp_file),
                    "-o",
                    str(png_file),
                    "-t",
                    "default",
                    "-b",
                    "white",
                    "--width",
                    "1200",
                    "--height",
                    "800",
                ],
                check=True,
                capture_output=True,
            )
            print(f"  ✅ PNG: {png_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ PNG export failed: {e}")
            success = False

        # Export to SVG (scalable for web)
        svg_file = output_dir / f"{title}.svg"
        try:
            subprocess.run(
                [
                    "mmdc",
                    "-i",
                    str(temp_file),
                    "-o",
                    str(svg_file),
                    "-t",
                    "default",
                    "-b",
                    "white",
                ],
                check=True,
                capture_output=True,
            )
            print(f"  ✅ SVG: {svg_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ SVG export failed: {e}")
            success = False

        return success

    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()


def main():
    """Main function."""
    print("🎨 Mermaid Diagram Export Tool")
    print("=" * 32)

    # Check if mermaid-cli is available
    if not check_mermaid_cli():
        if not install_mermaid_cli():
            sys.exit(1)

    # Setup paths
    script_dir = Path(__file__).parent
    mermaid_file = script_dir / "mermaid_diagrams.md"
    output_dir = script_dir / "presentation_diagrams" / "mermaid_exports"

    # Check input file
    if not mermaid_file.exists():
        print(f"❌ Input file not found: {mermaid_file}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Extract diagrams
    print(f"🔍 Processing: {mermaid_file.name}")
    diagrams = extract_mermaid_blocks(mermaid_file)

    if not diagrams:
        print("❌ No mermaid diagrams found!")
        sys.exit(1)

    print(f"📊 Found {len(diagrams)} diagrams")

    # Export each diagram
    success_count = 0
    for title, code, _section in diagrams:
        print(f"\n📈 Exporting: {title}")
        if export_diagram(title, code, output_dir):
            success_count += 1

    # Summary
    print("\n✅ Export complete!")
    print(f"📊 Successfully exported: {success_count}/{len(diagrams)} diagrams")
    print(f"📁 Files saved to: {output_dir}")

    # List generated files
    files = list(output_dir.glob("*"))
    if files:
        print("\n📋 Generated files:")
        for file in sorted(files):
            print(f"   - {file.name}")

    print("\n💡 Usage in presentations:")
    print("   ![Diagram](presentation_diagrams/mermaid_exports/diagram_name.png)")


if __name__ == "__main__":
    main()
