import unittest

from chicken_detector import count_chicken_detections


class ChickenDetectorCountTests(unittest.TestCase):
    def test_count_chicken_labels(self):
        detections = [
            {"label": "chicken", "confidence": 0.9},
            {"label": "hen", "confidence": 0.88},
            {"label": "bird", "confidence": 0.81},
        ]
        self.assertEqual(count_chicken_detections(detections), 2)

    def test_fallback_to_bird_when_no_chicken_label(self):
        detections = [
            {"label": "bird", "confidence": 0.9},
            {"label": "bird", "confidence": 0.8},
            {"label": "person", "confidence": 0.7},
        ]
        self.assertEqual(count_chicken_detections(detections), 2)

    def test_disable_bird_fallback(self):
        detections = [{"label": "bird", "confidence": 0.9}]
        self.assertEqual(count_chicken_detections(detections, allow_bird_fallback=False), 0)


if __name__ == "__main__":
    unittest.main()
