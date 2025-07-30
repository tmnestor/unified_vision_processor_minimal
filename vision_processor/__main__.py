#!/usr/bin/env python3
"""
Unified Vision Processor CLI Entry Point
=======================================

Single entry point for all vision processing commands:
python -m vision_processor <command>

This replaces the previous scattered CLI interfaces and provides a clean,
unified command structure based on the CLI consolidation plan.
"""

from .cli.unified_cli import app

if __name__ == "__main__":
    app()