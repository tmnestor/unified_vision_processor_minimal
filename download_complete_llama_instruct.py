#!/usr/bin/env python3
"""
Download complete Llama-3.2-11B-Vision-Instruct repository using Python.

This script downloads all files in the repository, with progress tracking
and resume capability.
"""

import os

from huggingface_hub import snapshot_download


def download_complete_llama():
    """Download the complete Llama-3.2-11B-Vision-Instruct repository."""

    # Set token
    os.environ["HF_TOKEN"] = ""

    print("🚀 Downloading complete Llama-3.2-11B-Vision-Instruct repository")
    print("📁 Destination: /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision-Instruct")
    print("⏱️  This will download ~22GB of files...")
    print("")

    try:
        # Download complete repository
        snapshot_download(
            repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
            local_dir="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision-Instruct",
            resume_download=True,  # Resume if interrupted
            local_files_only=False,  # Download from remote
            # Skip .git files to save space
            ignore_patterns=["*.git*", "*.gitattributes"],
        )

        print("\n✅ Download complete!")
        print("📁 Model location: /Users/tod/PretrainedLLM/Llama-3.2-11B-Vision-Instruct")
        print("🎯 Ready to test NER system!")

        # List downloaded files
        import pathlib

        model_dir = pathlib.Path("/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision-Instruct")
        if model_dir.exists():
            print("\n📋 Downloaded files:")
            for file in sorted(model_dir.rglob("*")):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  {file.name}: {size_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("💡 The download can be resumed by running this script again")
        return False


if __name__ == "__main__":
    success = download_complete_llama()
    if success:
        print("\n🎉 SUCCESS! Llama-3.2-11B-Vision-Instruct is ready!")
    else:
        print("\n💔 Download incomplete - try running again to resume")
