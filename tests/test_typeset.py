import cv2
import numpy as np
import pytest

from manga_translator.typeset import Typesetter, _break_lines
from manga_translator.utils import FONTS_DIR, natural_key

FONT = FONTS_DIR / "ComicNeue-Bold.ttf"


def ellipse(w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 1, h // 2 - 1), 0, 0, 360, 1, -1)
    return m.astype(bool)


@pytest.fixture
def ts():
    return Typesetter(FONT, min_px=12, max_px=40)


def test_all_lines_inside_shape(ts):
    mask = ellipse(220, 300)
    layout = ts.fit("I dipped into my savings a little, that's all.", mask)
    assert not layout.overflow
    for line in layout.lines:
        assert line.x >= 0 and line.x + line.width <= mask.shape[1]
        # Line must sit on rows that are inside the balloon at its left and right ends.
        y = int(line.y + layout.font_size * 0.6)
        assert mask[y, int(line.x) + 1] and mask[y, int(line.x + line.width) - 1]


def test_text_is_preserved(ts):
    text = "So they were in motion after all..."
    layout = ts.fit(text, ellipse(200, 200))
    assert " ".join(l.text for l in layout.lines) == text


def test_lozenge_shape_in_round_balloon(ts):
    layout = ts.fit("The faction that had been simmering under the unshakable regime finally rebelled.",
                    ellipse(260, 260))
    widths = [l.width for l in layout.lines]
    assert len(widths) >= 3
    # Middle lines are wider than the first and last in a round balloon.
    assert max(widths[1:-1]) > widths[0] and max(widths[1:-1]) > widths[-1]


def test_bigger_balloon_gets_bigger_text(ts):
    small = ts.fit("How did you get this?", ellipse(120, 160))
    big = ts.fit("How did you get this?", ellipse(260, 340))
    assert big.font_size > small.font_size


def test_falls_back_to_short_translation(ts):
    tiny = ellipse(60, 70)
    layout = ts.fit("This is an extremely long sentence that cannot possibly fit in here at all.", tiny,
                    short_text="No way.")
    assert layout.used_short and not layout.overflow


def test_overflow_is_flagged_not_dropped(ts):
    layout = ts.fit("Far too many words for such a tiny balloon to ever hold.", ellipse(40, 40))
    assert layout.overflow
    assert "balloon" in " ".join(l.text for l in layout.lines)


def test_avoids_ending_line_on_article():
    words = "I saw the cat on the mat".split()
    widths = [len(w) * 10 for w in words]
    _, groups = _break_lines(words, widths, 5, [120, 120])
    first_line_last_word = words[groups[0][1] - 1]
    assert first_line_last_word.lower() not in {"the", "on"}


def test_natural_page_order():
    names = ["page10.jpg", "page2.jpg", "page1.jpg", "page0.jpg"]
    assert sorted(names, key=natural_key) == ["page0.jpg", "page1.jpg", "page2.jpg", "page10.jpg"]
