from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

CHICKEN_KEYWORDS = {"chicken", "hen", "rooster", "cock", "cockerel", "chick"}


def _is_chicken_label(label: str) -> bool:
    normalized = label.strip().lower()
    return any(keyword in normalized for keyword in CHICKEN_KEYWORDS)


def count_chicken_detections(
    detections: Iterable[dict],
    allow_bird_fallback: bool = True,
) -> int:
    """Count chicken detections from YOLO class-name outputs.

    If a chicken-specific class is unavailable (e.g. COCO model),
    bird detections can be used as a fallback.
    """

    labels = [str(item.get("label", "")).strip().lower() for item in detections]
    chicken_count = sum(1 for label in labels if _is_chicken_label(label))
    if chicken_count > 0 or not allow_bird_fallback:
        return chicken_count
    return sum(1 for label in labels if label == "bird")


def detect_chickens_in_image(
    image_path: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    output_path: str | None = None,
    allow_bird_fallback: bool = True,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)
    if not results:
        return {"count": 0, "detections": []}

    result = results[0]
    boxes = result.boxes
    names = result.names or {}

    detections = []
    if boxes is not None and boxes.cls is not None and boxes.conf is not None:
        class_ids = boxes.cls.tolist()
        confidences = boxes.conf.tolist()
        for class_id, confidence in zip(class_ids, confidences):
            class_id_int = int(class_id)
            if isinstance(names, dict):
                label = names.get(class_id_int, str(class_id_int))
            else:
                label = names[class_id_int]
            detections.append({"label": str(label), "confidence": float(confidence)})

    chicken_count = count_chicken_detections(detections, allow_bird_fallback=allow_bird_fallback)

    saved_output_path = None
    if output_path:
        import cv2

        annotated = result.plot()
        cv2.imwrite(output_path, annotated)
        saved_output_path = str(Path(output_path).resolve())

    return {
        "count": chicken_count,
        "detections": detections,
        "output_path": saved_output_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and count chickens from a captured photo.")
    parser.add_argument("--image", required=True, help="Path to captured photo")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--output",
        default="detected_output.jpg",
        help="Path to save annotated detection image",
    )
    parser.add_argument(
        "--no-bird-fallback",
        action="store_true",
        help="Disable bird-as-chicken fallback for generic COCO models",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = detect_chickens_in_image(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        output_path=args.output,
        allow_bird_fallback=not args.no_bird_fallback,
    )

    print(f"chicken_count={result['count']}")
    if result.get("output_path"):
        print(f"saved_result={result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
