"""Simple Extraction Pipeline
============================

Implements a simple two-stage extraction pipeline:
1. Primary extraction from model response
2. AWK fallback for markdown content
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from rich.console import Console

from .patterns import PatternLibrary

console = Console()


class SimpleExtractionPipeline:
    """Simple extraction pipeline with primary + AWK fallback."""

    def __init__(self, awk_script_path: Optional[Path] = None, verbose: bool = False):
        """Initialize pipeline.

        Args:
            awk_script_path: Path to AWK markdown processor script
            verbose: Enable verbose output
        """
        self.awk_script_path = awk_script_path or Path("awk_markdown_processor.py")
        self.verbose = verbose
        self.pattern_library = PatternLibrary

    def extract(self, response: str, image_name: str) -> Tuple[Dict[str, str], str]:
        """Extract fields using primary extraction with AWK fallback.

        Args:
            response: Model response text
            image_name: Name of image being processed

        Returns:
            Tuple of (extracted_fields, extraction_method)
        """
        # Stage 1: Primary extraction
        fields = self._primary_extraction(response)

        if fields and len(fields) >= 3:  # Success threshold
            if self.verbose:
                console.print(
                    f"✅ Primary extraction successful: {len(fields)} fields",
                    style="green",
                )
            return fields, "primary"

        # Stage 2: AWK fallback for markdown
        if self._is_markdown_response(response):
            if self.verbose:
                console.print("🔄 Attempting AWK markdown fallback", style="yellow")

            awk_fields = self._awk_fallback(response, image_name)
            if awk_fields:
                if self.verbose:
                    console.print(
                        f"✅ AWK extraction successful: {len(awk_fields)} fields",
                        style="green",
                    )
                return awk_fields, "awk_fallback"

        # Return whatever we got from primary extraction
        if self.verbose:
            console.print(
                f"⚠️  Limited extraction: {len(fields)} fields", style="yellow"
            )
        return fields, "primary_limited"

    def _primary_extraction(self, response: str) -> Dict[str, str]:
        """Primary field extraction using patterns."""
        fields = {}

        # Try structured key-value extraction
        kv_pattern = re.compile(
            r"^([A-Z][A-Z\s_/-]+?):\s*(.+)$", re.MULTILINE | re.IGNORECASE
        )
        matches = kv_pattern.findall(response)

        for key, value in matches:
            clean_key = key.strip().replace(" ", "_").upper()
            clean_value = value.strip()
            if (
                clean_key
                and clean_value
                and clean_value.upper() not in ["N/A", "NONE", "NULL"]
            ):
                fields[clean_key] = clean_value

        # Try pattern-based extraction for common fields
        if "DATE" not in fields:
            date = self.pattern_library.extract_date(response)
            if date:
                fields["DATE"] = date

        if "TOTAL" not in fields:
            total = self.pattern_library.extract_currency(response)
            if total:
                fields["TOTAL"] = total

        if "ABN" not in fields:
            abn = self.pattern_library.extract_abn(response)
            if abn:
                fields["ABN"] = abn

        return fields

    def _is_markdown_response(self, response: str) -> bool:
        """Check if response contains markdown formatting."""
        markdown_indicators = [
            r"\|.*\|.*\|",  # Table
            r"^#+\s",  # Headers
            r"^\*\s",  # Bullets
            r"\*\*[^*]+\*\*",  # Bold
        ]

        for pattern in markdown_indicators:
            if re.search(pattern, response, re.MULTILINE):
                return True
        return False

    def _awk_fallback(self, response: str, image_name: str) -> Dict[str, str]:
        """Use AWK script to process markdown content."""
        if not self.awk_script_path.exists():
            if self.verbose:
                console.print(
                    f"❌ AWK script not found: {self.awk_script_path}", style="red"
                )
            return {}

        try:
            # Write response to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            ) as tmp:
                tmp.write(response)
                tmp_path = tmp.name

            # Run AWK processor
            result = subprocess.run(
                ["python", str(self.awk_script_path), tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Parse AWK output
                return self._parse_awk_output(result.stdout)
            else:
                if self.verbose:
                    console.print(
                        f"❌ AWK processing failed: {result.stderr}", style="red"
                    )

        except Exception as e:
            if self.verbose:
                console.print(f"❌ AWK fallback error: {e}", style="red")

        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

        return {}

    def _parse_awk_output(self, output: str) -> Dict[str, str]:
        """Parse AWK processor output into fields."""
        fields = {}

        # AWK output is typically key: value format
        lines = output.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                clean_key = key.strip().upper().replace(" ", "_")
                clean_value = value.strip()
                if clean_key and clean_value:
                    fields[clean_key] = clean_value

        return fields
