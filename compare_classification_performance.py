#!/usr/bin/env python3
"""Compare Document Classification Performance between InternVL3 and Llama-3.2-Vision

This script tests both models on the same set of documents to compare:
- Classification accuracy
- Confidence scores
- Processing speed
- Document type recognition capabilities
"""

import json
import random
import time
from pathlib import Path

from vision_processor.config.model_factory import ModelFactory
from vision_processor.config.simple_config import SimpleConfig


def test_classification_performance():
    """Compare classification performance between InternVL3 and Llama-3.2-Vision."""
    print("=== DOCUMENT CLASSIFICATION PERFORMANCE COMPARISON ===")
    print("InternVL3 vs Llama-3.2-Vision\n")

    # Sample diverse images for testing (mix of different types)
    datasets_path = Path("datasets")
    all_images = list(datasets_path.glob("*.png"))

    # Select a representative sample for testing
    sample_images = [
        "image14.png",  # Receipt (known working)
        "image1.png",  # First image
        "image10.png",  # Different document
        "image25.png",  # Mid-range sample
        "image50.png",  # Mid-range sample
        "image75.png",  # Higher numbered sample
        "image100.png",  # High numbered sample
    ]

    # Ensure all sample images exist
    test_images = []
    for img_name in sample_images:
        img_path = datasets_path / img_name
        if img_path.exists():
            test_images.append(str(img_path))
        else:
            print(f"Warning: {img_name} not found, skipping")

    # If we have fewer than 5 test images, add random ones
    if len(test_images) < 5:
        random_images = random.sample(all_images, min(5, len(all_images)))
        test_images.extend([str(img) for img in random_images if str(img) not in test_images])

    test_images = test_images[:7]  # Limit to 7 for focused comparison

    print(f"Testing {len(test_images)} documents:")
    for i, img in enumerate(test_images, 1):
        print(f"  {i}. {Path(img).name}")
    print()

    results = {
        "internvl3": [],
        "llama32_vision": [],
        "comparison": {
            "total_documents": len(test_images),
            "agreement_count": 0,
            "disagreement_count": 0,
            "internvl3_errors": 0,
            "llama32_errors": 0,
            "performance_summary": {},
        },
    }

    # Test InternVL3
    print("=== TESTING INTERNVL3 ===")
    config_internvl = SimpleConfig()
    config_internvl.model_type = "internvl3"
    config_internvl.model_path = "/home/jovyan/nfs_share/models/InternVL3-8B"

    try:
        model_internvl = ModelFactory.create_model(config_internvl)

        internvl_total_time = 0
        for i, image_path in enumerate(test_images, 1):
            print(f"InternVL3 - Document {i}/{len(test_images)}: {Path(image_path).name}")

            start_time = time.time()
            try:
                classification_result = model_internvl.classify_document(image_path)
                processing_time = time.time() - start_time
                internvl_total_time += processing_time

                result = {
                    "image": Path(image_path).name,
                    "document_type": classification_result["document_type"],
                    "confidence": classification_result["confidence"],
                    "is_business_document": classification_result["is_business_document"],
                    "processing_time": processing_time,
                    "classification_response": classification_result["classification_response"][:200]
                    + "..."
                    if len(classification_result["classification_response"]) > 200
                    else classification_result["classification_response"],
                    "success": True,
                }

                print(f"  Type: {result['document_type']} (confidence: {result['confidence']:.2f})")
                print(f"  Time: {processing_time:.2f}s")
                print(f"  Business doc: {result['is_business_document']}")

            except Exception as e:
                processing_time = time.time() - start_time
                result = {
                    "image": Path(image_path).name,
                    "document_type": "error",
                    "confidence": 0.0,
                    "is_business_document": False,
                    "processing_time": processing_time,
                    "classification_response": f"Error: {str(e)}",
                    "success": False,
                }
                results["comparison"]["internvl3_errors"] += 1
                print(f"  ERROR: {str(e)}")

            results["internvl3"].append(result)
            print()

        print(
            f"InternVL3 total time: {internvl_total_time:.2f}s (avg: {internvl_total_time / len(test_images):.2f}s per doc)"
        )

    except Exception as e:
        print(f"Failed to load InternVL3: {e}")
        results["comparison"]["internvl3_errors"] = len(test_images)

    print("\n" + "=" * 60 + "\n")

    # Test Llama-3.2-Vision
    print("=== TESTING LLAMA-3.2-VISION ===")
    config_llama = SimpleConfig()
    config_llama.model_type = "llama32_vision"
    config_llama.model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"

    try:
        model_llama = ModelFactory.create_model(config_llama)

        llama_total_time = 0
        for i, image_path in enumerate(test_images, 1):
            print(f"Llama-3.2-Vision - Document {i}/{len(test_images)}: {Path(image_path).name}")

            start_time = time.time()
            try:
                classification_result = model_llama.classify_document(image_path)
                processing_time = time.time() - start_time
                llama_total_time += processing_time

                result = {
                    "image": Path(image_path).name,
                    "document_type": classification_result["document_type"],
                    "confidence": classification_result["confidence"],
                    "is_business_document": classification_result["is_business_document"],
                    "processing_time": processing_time,
                    "classification_response": classification_result["classification_response"][:200]
                    + "..."
                    if len(classification_result["classification_response"]) > 200
                    else classification_result["classification_response"],
                    "success": True,
                }

                print(f"  Type: {result['document_type']} (confidence: {result['confidence']:.2f})")
                print(f"  Time: {processing_time:.2f}s")
                print(f"  Business doc: {result['is_business_document']}")

            except Exception as e:
                processing_time = time.time() - start_time
                result = {
                    "image": Path(image_path).name,
                    "document_type": "error",
                    "confidence": 0.0,
                    "is_business_document": False,
                    "processing_time": processing_time,
                    "classification_response": f"Error: {str(e)}",
                    "success": False,
                }
                results["comparison"]["llama32_errors"] += 1
                print(f"  ERROR: {str(e)}")

            results["llama32_vision"].append(result)
            print()

        print(
            f"Llama-3.2-Vision total time: {llama_total_time:.2f}s (avg: {llama_total_time / len(test_images):.2f}s per doc)"
        )

    except Exception as e:
        print(f"Failed to load Llama-3.2-Vision: {e}")
        results["comparison"]["llama32_errors"] = len(test_images)

    print("\n" + "=" * 60 + "\n")

    # Generate comparison analysis
    print("=== DETAILED COMPARISON ANALYSIS ===")

    # Side-by-side comparison
    print("\nSIDE-BY-SIDE RESULTS:")
    print(f"{'Document':<15} {'InternVL3':<20} {'Llama-3.2':<20} {'Agreement':<12} {'Speed Winner'}")
    print("-" * 85)

    agreement_count = 0
    for i in range(len(test_images)):
        if i < len(results["internvl3"]) and i < len(results["llama32_vision"]):
            internvl_result = results["internvl3"][i]
            llama_result = results["llama32_vision"][i]

            doc_name = internvl_result["image"][:12] + "..."
            internvl_type = internvl_result["document_type"][:15]
            llama_type = llama_result["document_type"][:15]

            agreement = (
                "✓ MATCH"
                if internvl_result["document_type"] == llama_result["document_type"]
                else "✗ DIFFER"
            )
            if agreement == "✓ MATCH":
                agreement_count += 1

            # Speed comparison
            if internvl_result["success"] and llama_result["success"]:
                if internvl_result["processing_time"] < llama_result["processing_time"]:
                    speed_winner = "InternVL3"
                elif llama_result["processing_time"] < internvl_result["processing_time"]:
                    speed_winner = "Llama-3.2"
                else:
                    speed_winner = "Tie"
            else:
                speed_winner = "N/A"

            print(f"{doc_name:<15} {internvl_type:<20} {llama_type:<20} {agreement:<12} {speed_winner}")

    results["comparison"]["agreement_count"] = agreement_count
    results["comparison"]["disagreement_count"] = len(test_images) - agreement_count

    # Performance summary
    print("\nPERFORMANCE SUMMARY:")
    print(
        f"Agreement Rate: {agreement_count}/{len(test_images)} ({agreement_count / len(test_images) * 100:.1f}%)"
    )
    print(f"InternVL3 Errors: {results['comparison']['internvl3_errors']}")
    print(f"Llama-3.2 Errors: {results['comparison']['llama32_errors']}")

    # Calculate average confidence scores
    internvl_confidences = [r["confidence"] for r in results["internvl3"] if r["success"]]
    llama_confidences = [r["confidence"] for r in results["llama32_vision"] if r["success"]]

    if internvl_confidences:
        avg_internvl_confidence = sum(internvl_confidences) / len(internvl_confidences)
        print(f"InternVL3 Avg Confidence: {avg_internvl_confidence:.2f}")

    if llama_confidences:
        avg_llama_confidence = sum(llama_confidences) / len(llama_confidences)
        print(f"Llama-3.2 Avg Confidence: {avg_llama_confidence:.2f}")

    # Calculate average processing times
    internvl_times = [r["processing_time"] for r in results["internvl3"] if r["success"]]
    llama_times = [r["processing_time"] for r in results["llama32_vision"] if r["success"]]

    if internvl_times:
        avg_internvl_time = sum(internvl_times) / len(internvl_times)
        print(f"InternVL3 Avg Time: {avg_internvl_time:.2f}s")

    if llama_times:
        avg_llama_time = sum(llama_times) / len(llama_times)
        print(f"Llama-3.2 Avg Time: {avg_llama_time:.2f}s")

    # Document type distribution
    print("\nDOCUMENT TYPE DISTRIBUTION:")
    internvl_types = {}
    llama_types = {}

    for result in results["internvl3"]:
        if result["success"]:
            doc_type = result["document_type"]
            internvl_types[doc_type] = internvl_types.get(doc_type, 0) + 1

    for result in results["llama32_vision"]:
        if result["success"]:
            doc_type = result["document_type"]
            llama_types[doc_type] = llama_types.get(doc_type, 0) + 1

    print("InternVL3 classifications:")
    for doc_type, count in sorted(internvl_types.items()):
        print(f"  {doc_type}: {count}")

    print("Llama-3.2 classifications:")
    for doc_type, count in sorted(llama_types.items()):
        print(f"  {doc_type}: {count}")

    # Save detailed results to JSON
    results_file = Path("classification_comparison_results.json")
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    # Final recommendations
    print("\n=== RECOMMENDATIONS ===")

    if agreement_count / len(test_images) > 0.7:
        print("✓ High agreement between models suggests reliable classification")
    else:
        print("⚠ Low agreement suggests models may have different strengths")

    if internvl_confidences and llama_confidences:
        if avg_internvl_confidence > avg_llama_confidence:
            print("✓ InternVL3 shows higher confidence scores")
        elif avg_llama_confidence > avg_internvl_confidence:
            print("✓ Llama-3.2-Vision shows higher confidence scores")
        else:
            print("≈ Similar confidence levels between models")

    if internvl_times and llama_times:
        if avg_internvl_time < avg_llama_time:
            print("⚡ InternVL3 is faster for classification")
        elif avg_llama_time < avg_internvl_time:
            print("⚡ Llama-3.2-Vision is faster for classification")
        else:
            print("≈ Similar processing speeds")

    if results["comparison"]["internvl3_errors"] == 0 and results["comparison"]["llama32_errors"] > 0:
        print("✓ InternVL3 shows better stability (fewer errors)")
    elif results["comparison"]["llama32_errors"] == 0 and results["comparison"]["internvl3_errors"] > 0:
        print("✓ Llama-3.2-Vision shows better stability (fewer errors)")
    elif results["comparison"]["internvl3_errors"] == 0 and results["comparison"]["llama32_errors"] == 0:
        print("✓ Both models show excellent stability")


if __name__ == "__main__":
    test_classification_performance()
