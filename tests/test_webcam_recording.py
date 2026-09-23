"""Verify recorded video keeps wall-clock duration when detection is slower."""

import unittest

import numpy as np

from perception.webcam import write_timed_frame


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(int(frame[0, 0, 0]))


class TimedRecordingTests(unittest.TestCase):
    def test_repeats_previous_frame_to_preserve_elapsed_time(self) -> None:
        writer = FakeWriter()
        previous = None
        count = 0
        for value, elapsed in ((1, 0.0), (2, 0.1), (3, 0.2)):
            frame = np.full((1, 1, 3), value, dtype=np.uint8)
            previous, count = write_timed_frame(writer, frame, previous, elapsed, 30.0, count)
        self.assertEqual(writer.frames, [1, 1, 1, 2, 2, 2, 3])
        self.assertEqual(count / 30.0, 7 / 30.0)


if __name__ == "__main__":
    unittest.main()
