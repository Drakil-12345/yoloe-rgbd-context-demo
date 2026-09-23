import unittest
from perception.common import LIBRARY_PATH
from perception.prompts import PromptOptimizer


class PromptOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = PromptOptimizer(LIBRARY_PATH)

    def assert_resolution(self, text, concept, prompt, color=None, spatial=None, selection=None):
        result = self.optimizer.resolve(text)
        self.assertEqual(result.concept, concept)
        self.assertEqual(result.optimized_prompt, prompt)
        self.assertEqual(result.color, color)
        self.assertEqual(result.spatial, spatial or [])
        self.assertEqual(result.selection, selection)

    def test_vietnamese_context_is_separated_from_prompt(self):
        self.assert_resolution(
            "Tìm chai nước màu xanh ở bên phải",
            "water_bottle",
            "bottle",
            color="blue",
            spatial=["right"],
        )

    def test_unaccented_stt(self):
        self.assert_resolution("tim chai nuoc ben trai", "water_bottle", "bottle", spatial=["left"])

    def test_english_water_bottle(self):
        self.assert_resolution("find the blue water bottle", "water_bottle", "bottle", color="blue")

    def test_fuzzy_asr_typo(self):
        result = self.optimizer.resolve("tim chai nuok ben phai")
        self.assertEqual(result.concept, "water_bottle")
        self.assertEqual(result.match_type, "fuzzy_asr")

    def test_cup(self):
        self.assert_resolution("cái ly đỏ ở bên trái", "cup", "cup", color="red", spatial=["left"])

    def test_phone(self):
        self.assert_resolution("dien thoai ben phai", "cell_phone", "cell phone", spatial=["right"])

    def test_laptop_center(self):
        self.assert_resolution("máy tính xách tay ở giữa", "laptop", "laptop", spatial=["center"])

    def test_largest_chair(self):
        self.assert_resolution("find the largest chair", "chair", "chair", selection="largest")

    def test_remote(self):
        self.assert_resolution("tìm điều khiển từ xa", "remote_control", "remote control")

    def test_word_boundary_prevents_chairman_match(self):
        self.assertIsNone(self.optimizer.resolve("find the chairman").concept)

    def test_negated_object_is_excluded(self):
        result = self.optimizer.resolve("đừng tìm chai nước, tìm cái cốc đỏ")
        self.assertEqual(result.concept, "cup")
        self.assertEqual(result.color, "red")
        self.assertEqual(result.excluded_concepts, ["water_bottle"])

    def test_only_negation_does_not_run_object(self):
        result = self.optimizer.resolve("không phải chai nước")
        self.assertIsNone(result.concept)
        self.assertEqual(result.excluded_concepts, ["water_bottle"])

    def test_unknown_object(self):
        self.assertIsNone(self.optimizer.resolve("tìm quả táo").concept)


if __name__ == "__main__":
    unittest.main()
