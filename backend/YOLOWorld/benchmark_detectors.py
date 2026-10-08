"""
Benchmark script to compare YOLOE with other detectors.

Usage:
    python benchmark_detectors.py --image path/to/image.jpg --prompt "person . car"
"""

import sys
import time
import argparse
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import cv2
import numpy as np
from app.core.config import settings
from app.core.logging import logger


def benchmark_detector(detector_name: str, image: np.ndarray, prompt: str, iterations: int = 3):
    """Benchmark a single detector."""

    # Set detector
    original_model = settings.DETECTION_MODEL
    settings.DETECTION_MODEL = detector_name

    try:
        # Import and load detector
        from app.services.detector import get_detector

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {detector_name.upper()}")
        logger.info(f"{'='*60}")

        # Load detector (timed)
        load_start = time.time()
        detector = get_detector()
        load_time = time.time() - load_start
        logger.info(f"Load time: {load_time:.2f}s")

        # Warm-up run
        logger.info("Warming up...")
        _ = detector.predict(image, prompt, 0.25, 0.25)

        # Benchmark runs
        times = []
        all_detections = []

        for i in range(iterations):
            start = time.time()
            detections = detector.predict(image, prompt, 0.25, 0.25)
            elapsed = time.time() - start
            times.append(elapsed)
            all_detections.append(detections)
            logger.info(f"Run {i+1}/{iterations}: {elapsed:.3f}s, {len(detections)} detections")

        # Statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_detections = np.mean([len(d) for d in all_detections])

        logger.info(f"\nResults:")
        logger.info(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        logger.info(f"  Average detections: {avg_detections:.1f}")
        logger.info(f"  FPS: {1/avg_time:.2f}")

        return {
            'detector': detector_name,
            'load_time': load_time,
            'avg_time': avg_time,
            'std_time': std_time,
            'avg_detections': avg_detections,
            'fps': 1/avg_time,
            'detections': all_detections[0],  # First run detections
        }

    except Exception as e:
        logger.error(f"Failed to benchmark {detector_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Restore original setting
        settings.DETECTION_MODEL = original_model

        # Clear detector instance for next benchmark
        from app.services import detector as detector_module
        detector_module._detector_instance = None


def main():
    parser = argparse.ArgumentParser(description="Benchmark detection models")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person . car . dog",
        help="Detection prompt"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per detector"
    )
    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",
        default=["grounding_dino", "yoloe"],
        help="Detectors to benchmark (grounding_dino, florence2, yoloe)"
    )

    args = parser.parse_args()

    # Load image
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image: {args.image}")
        return 1

    h, w = image.shape[:2]
    logger.info(f"\nImage: {args.image}")
    logger.info(f"Size: {w}x{h}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Iterations: {args.iterations}")

    # Benchmark each detector
    results = []
    for detector_name in args.detectors:
        result = benchmark_detector(detector_name, image, args.prompt, args.iterations)
        if result:
            results.append(result)

    # Print comparison table
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Detector':<20} {'Load (s)':<12} {'Inference (s)':<15} {'FPS':<10} {'Detections':<12}")
        print(f"{'-'*80}")

        for r in results:
            print(f"{r['detector']:<20} {r['load_time']:<12.2f} "
                  f"{r['avg_time']:.3f} ± {r['std_time']:.3f}    "
                  f"{r['fps']:<10.2f} {r['avg_detections']:<12.1f}")

        # Find fastest
        fastest = min(results, key=lambda x: x['avg_time'])
        print(f"\n✓ Fastest: {fastest['detector']} ({fastest['fps']:.2f} FPS)")

        # Find most detections
        most_dets = max(results, key=lambda x: x['avg_detections'])
        print(f"✓ Most detections: {most_dets['detector']} ({most_dets['avg_detections']:.1f} objects)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
