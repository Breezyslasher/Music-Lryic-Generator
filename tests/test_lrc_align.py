import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lrc_align as la  # noqa: E402

LINE_FILE = """[ar:Someone]
[00:15.23] Oh, oh, oh oh oh
[00:18.59]
[00:22.86] I guarantee you'd keep it secret

[00:26.72][01:10.00] So give it to me now
"""

WORD_FILE = """[00:28.90]<00:28.90>I <00:29.34>got <00:29.62>a <00:29.81>feeling
[00:36.66]<00:36.66>That <00:37.06>tonight's <00:37.79>gonna <00:38.32>be
"""


def fake_align(words_per_second=2.0):
    """Aligner that places words evenly from the window start."""
    def fn(start, end, text):
        out = []
        t = start + la.LEAD_PAD
        for w in text.split():
            out.append((w, t, t + 1 / words_per_second, 0.9))
            t += 1 / words_per_second
        return out
    return fn


def context_align(lines, offset=0.0, words_per_second=2.0):
    """Aligner that knows the songs: each text line's words start at that
    line's tag plus ``offset``, like a good alignment would find them."""
    by_text = {}
    for ln in lines:
        if ln.is_lyric:
            by_text.setdefault(la.clean_text(ln), []).append(ln.start)

    def fn(start, end, text):
        out = []
        for piece in text.split("\n"):
            starts = by_text.get(piece.strip(), [])
            inside = [s for s in starts if start - 1 <= s <= end + 1]
            t = (inside or starts or [start])[0] + offset
            for w in piece.split():
                out.append((w, t, t + 1 / words_per_second, 0.9))
                t += 1 / words_per_second
        return out
    return fn


class TimestampTests(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertAlmostEqual(la.parse_timestamp("01", "05", "23"), 65.23)
        self.assertAlmostEqual(la.parse_timestamp("0", "5", "905"), 5.905)
        self.assertEqual(la.format_timestamp(65.23), "01:05.23")
        self.assertEqual(la.format_timestamp(5.905, 3), "00:05.905")
        self.assertEqual(la.format_timestamp(0.999), "00:00.99")
        self.assertEqual(la.format_timestamp(-1), "00:00.00")


class ParseTests(unittest.TestCase):
    def test_parse_line_file(self):
        lines = la.parse_lrc(LINE_FILE)
        self.assertEqual(len(lines), 7)  # multi-timestamp line expanded into two
        self.assertIsNone(lines[0].start)  # metadata
        self.assertEqual(lines[1].tag, "[00:15.23]")
        self.assertEqual(lines[1].text, "Oh, oh, oh oh oh")
        self.assertTrue(lines[1].needs_alignment)
        self.assertAlmostEqual(lines[2].start, 18.59)
        self.assertFalse(lines[2].is_lyric)  # instrumental marker
        self.assertIsNone(lines[4].start)  # blank
        self.assertEqual(lines[5].raw, "[00:26.72]So give it to me now")
        self.assertAlmostEqual(lines[6].start, 70.0)
        self.assertFalse(la.is_word_level(lines))
        self.assertEqual(la.detect_decimals(lines), 2)

    def test_parse_word_file(self):
        lines = la.parse_lrc(WORD_FILE)
        self.assertTrue(all(ln.has_word_tags for ln in lines if ln.is_lyric))
        self.assertTrue(la.is_word_level(lines))
        self.assertFalse(any(ln.needs_alignment for ln in lines))

    def test_mixed_file_is_not_word_level(self):
        lines = la.parse_lrc(WORD_FILE + "[00:40.00] plain line\n")
        self.assertFalse(la.is_word_level(lines))
        self.assertTrue(la.has_any_word_tags(lines))

    def test_millisecond_tags(self):
        lines = la.parse_lrc("[00:00.905] Na-na-na\n[00:04.307] Na-na-na\n")
        self.assertAlmostEqual(lines[0].start, 0.905)
        self.assertEqual(la.detect_decimals(lines), 3)

    def test_bracket_text_is_lyric(self):
        lines = la.parse_lrc("[00:00.19][Missy Elliott:]\n")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "[Missy Elliott:]")
        self.assertTrue(lines[0].is_lyric)

    def test_untimed_text_is_kept_untimed(self):
        lines = la.parse_lrc("Just some text\n")
        self.assertIsNone(lines[0].start)
        self.assertFalse(la.is_word_level(lines))


class WindowTests(unittest.TestCase):
    def test_windows_use_next_timed_line(self):
        lines = la.parse_lrc(LINE_FILE)
        windows = la.line_windows(lines, audio_duration=200.0)
        self.assertEqual(windows[1], (15.23, 18.59))
        self.assertEqual(windows[3], (22.86, 26.72))
        self.assertEqual(windows[5], (26.72, 26.72 + la.MAX_LINE_WINDOW))  # next is 70s away, capped
        self.assertEqual(windows[6], (70.0, 100.0))
        self.assertNotIn(2, windows)  # instrumental marker
        self.assertNotIn(0, windows)  # metadata

    def test_window_capped_by_audio_duration(self):
        lines = la.parse_lrc("[03:00.00] last line\n")
        windows = la.line_windows(lines, audio_duration=185.0)
        self.assertEqual(windows[0], (180.0, 185.0))

    def test_duplicate_timestamps_share_window(self):
        lines = la.parse_lrc("[00:10.00] a b\n[00:10.00] c d\n[00:14.00] e\n")
        windows = la.line_windows(lines, None)
        self.assertEqual(windows[0], (10.0, 14.0))
        self.assertEqual(windows[1], (10.0, 14.0))


class WordMappingTests(unittest.TestCase):
    def test_one_to_one(self):
        tokens = ["I", "got", "a", "feeling"]
        aligned = [("I", 1.0, 1.2, .9), ("got", 1.3, 1.5, .9), ("a", 1.6, 1.7, .9), ("feeling", 1.8, 2.5, .9)]
        self.assertEqual(la.map_aligned_words_to_tokens(tokens, aligned), [1.0, 1.3, 1.6, 1.8])

    def test_aligner_splits_differently(self):
        # Aligner glued punctuation and split a hyphenated token.
        tokens = ["Oh,", "oh", "5-4-3-2", "one"]
        aligned = [(" Oh", 1.0, 1.1, .9), (",", 1.1, 1.1, .9), (" oh", 1.2, 1.3, .9),
                   (" 5", 1.4, 1.5, .9), ("-4", 1.5, 1.6, .9), ("-3", 1.6, 1.7, .9), ("-2", 1.7, 1.8, .9),
                   (" one", 1.9, 2.0, .9)]
        self.assertEqual(la.map_aligned_words_to_tokens(tokens, aligned), [1.0, 1.2, 1.4, 1.9])

    def test_failed_words_become_none(self):
        tokens = ["a", "b"]
        aligned = [("a", 1.0, 1.2, .9), ("b", 9.0, 9.0, 0.0)]
        self.assertEqual(la.map_aligned_words_to_tokens(tokens, aligned), [1.0, None])

    def test_empty_alignment(self):
        self.assertEqual(la.map_aligned_words_to_tokens(["a", "b"], []), [None, None])


class FinalizeTests(unittest.TestCase):
    def test_first_word_pinned_and_monotonic(self):
        times, fb = la.finalize_word_times([1.5, 1.0, 1.9, 1.85], 1.2, 3.0)
        self.assertFalse(fb)
        self.assertEqual(times[0], 1.2)
        for a, b in zip(times, times[1:]):
            self.assertGreater(b, a)
        self.assertLessEqual(times[-1], 3.0)

    def test_gap_is_interpolated(self):
        times, fb = la.finalize_word_times([1.0, None, None, 4.0], 1.0, 10.0)
        self.assertFalse(fb)
        self.assertEqual(times, [1.0, 2.0, 3.0, 4.0])

    def test_fallback_when_mostly_missing(self):
        times, fb = la.finalize_word_times([None, None, 2.0, None], 1.0, 5.0)
        self.assertTrue(fb)
        self.assertEqual(times[0], 1.0)
        self.assertEqual(len(times), 4)
        self.assertLess(times[-1], 5.0)

    def test_fallback_respects_short_window(self):
        times = la.fallback_word_times(10, 1.0, 2.0)
        self.assertEqual(times[0], 1.0)
        self.assertLess(times[-1], 2.0)

    def test_clamped_to_window(self):
        times, _ = la.finalize_word_times([1.0, 8.0, 9.0], 1.0, 2.0)
        self.assertTrue(all(t < 2.0 for t in times))
        self.assertLess(times[1], times[2])
        # Overflowing words are spread through the gap, not stacked 0.01s apart.
        self.assertGreater(times[2] - times[1], 0.3)

    def test_tied_words_are_spread_forward(self):
        times, _ = la.finalize_word_times([1.0, 1.0, 1.0, 2.0], 1.0, 3.0)
        self.assertEqual([round(t, 3) for t in times], [1.0, 1.333, 1.667, 2.0])

    def test_words_before_line_start_are_spread(self):
        times, _ = la.finalize_word_times([0.2, 0.5, 0.9, 2.0], 1.0, 3.0)
        self.assertEqual([round(t, 3) for t in times], [1.0, 1.333, 1.667, 2.0])

    def test_trailing_words_past_window_are_spread_backward(self):
        times, _ = la.finalize_word_times([1.0, 5.0, 5.0, 5.0], 1.0, 2.0)
        self.assertEqual([round(t, 3) for t in times], [1.0, 1.33, 1.66, 1.99])

    def test_out_of_order_words_are_smoothed(self):
        times, _ = la.finalize_word_times([1.0, 1.9, 1.5, 2.5], 1.0, 4.0)
        self.assertEqual([round(t, 3) for t in times], [1.0, 1.7, 2.1, 2.5])
        for a, b in zip(times, times[1:]):
            self.assertGreater(b, a)

    def test_pool_adjacent_violators(self):
        self.assertEqual(la.pool_adjacent_violators([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
        self.assertEqual(la.pool_adjacent_violators([3, 2, 1]), [2, 2, 2])
        self.assertEqual(la.pool_adjacent_violators([1, 2, 3]), [1, 2, 3])


class ConvertTests(unittest.TestCase):
    def test_convert_lines(self):
        lines = la.parse_lrc(LINE_FILE)
        out, stats = la.convert_lines(lines, context_align(lines), audio_duration=200.0)
        self.assertEqual(out[0], "[ar:Someone]")
        self.assertEqual(out[2], "[00:18.59]")
        self.assertEqual(out[4], "")
        self.assertEqual(out[1], "[00:15.23]<00:15.23>Oh, <00:15.73>oh, <00:16.23>oh <00:16.73>oh <00:17.23>oh")
        self.assertTrue(out[6].startswith("[01:10.00]<01:10.00>So <01:10.50>give"))
        self.assertEqual(stats.lines_aligned, 4)
        self.assertEqual(stats.lines_fallback, 0)
        self.assertEqual(stats.lines_kept, 3)
        # The converted output is now detected as word-by-word.
        self.assertTrue(la.is_word_level(la.parse_lrc("\n".join(out))))

    def test_convert_uses_fallback_when_aligner_fails(self):
        lines = la.parse_lrc("[00:10.00] one two three\n[00:12.00] four\n")
        out, stats = la.convert_lines(lines, lambda s, e, t: [], audio_duration=None)
        self.assertEqual(stats.lines_fallback, 2)
        self.assertTrue(out[0].startswith("[00:10.00]<00:10.00>one <00:10.45>two <00:10.90>three"))

    def test_convert_survives_aligner_exception(self):
        def boom(s, e, t):
            raise RuntimeError("nope")
        lines = la.parse_lrc("[00:10.00] one two\n")
        out, stats = la.convert_lines(lines, boom)
        self.assertEqual(stats.lines_fallback, 1)
        self.assertIn("<00:10.00>one", out[0])

    def test_word_lines_are_kept_in_mixed_file(self):
        lines = la.parse_lrc(WORD_FILE + "[00:40.00] plain line\n")
        out, stats = la.convert_lines(lines, context_align(lines))
        self.assertEqual(out[0], WORD_FILE.splitlines()[0])
        self.assertEqual(out[2], "[00:40.00]<00:40.00>plain <00:40.50>line")
        self.assertEqual(stats.lines_aligned, 1)
        self.assertEqual(stats.lines_kept, 2)

    def test_millisecond_precision_preserved(self):
        lines = la.parse_lrc("[00:00.905] Na na\n")
        out, _ = la.convert_lines(lines, context_align(lines), decimals=la.detect_decimals(lines))
        self.assertEqual(out[0], "[00:00.905]<00:00.905>Na <00:01.405>na")


class ContextTests(unittest.TestCase):
    THREE = la.parse_lrc("[00:10.00] one two three\n[00:14.00] four five\n[00:17.00] six\n")

    def test_middle_line_gets_both_neighbours(self):
        ctx = la.build_context(self.THREE, 1, 17.0, 200.0)
        self.assertEqual(ctx.text, "one two three\nfour five\nsix")
        self.assertEqual(ctx.first_token, 3)
        self.assertTrue(ctx.has_prev)
        self.assertAlmostEqual(ctx.slice_start, 10.0 - la.LEAD_PAD)
        self.assertAlmostEqual(ctx.slice_end, 17.0 + la.CONTEXT_TAIL)

    def test_first_line_has_no_prev(self):
        ctx = la.build_context(self.THREE, 0, 14.0, 200.0)
        self.assertEqual(ctx.text, "one two three\nfour five")
        self.assertEqual(ctx.first_token, 0)
        self.assertFalse(ctx.has_prev)
        self.assertAlmostEqual(ctx.slice_start, 10.0 - la.LEAD_PAD_SOLO)
        self.assertAlmostEqual(ctx.slice_end, 17.0)  # capped at the line after next

    def test_last_line_has_no_next(self):
        ctx = la.build_context(self.THREE, 2, 17.0 + la.MAX_LINE_WINDOW, 20.0)
        self.assertEqual(ctx.text, "four five\nsix")
        self.assertEqual(ctx.first_token, 2)
        self.assertAlmostEqual(ctx.slice_end, 20.0)
        ctx = la.build_context(self.THREE, 2, 17.0 + la.MAX_LINE_WINDOW, 18.0)
        self.assertAlmostEqual(ctx.slice_end, 18.0)  # audio ends

    def test_far_neighbours_are_not_context(self):
        lines = la.parse_lrc("[00:10.00] one\n[00:30.00] two\n[00:50.00] three\n")
        ctx = la.build_context(lines, 1, 50.0, 200.0)
        self.assertEqual(ctx.text, "two")
        self.assertFalse(ctx.has_prev)
        self.assertAlmostEqual(ctx.slice_end, 30.0 + la.SOLO_TAIL_MAX)

    def test_lone_long_line_gets_more_audio(self):
        lines = la.parse_lrc("[00:10.00] " + " ".join(["w"] * 40) + "\n")
        ctx = la.build_context(lines, 0, 40.0, 200.0)
        self.assertAlmostEqual(ctx.slice_end, 10.0 + la.SOLO_TAIL_MAX)


class SpanCapTests(unittest.TestCase):
    def test_reasonable_span_untouched(self):
        times = [10.0, 10.3, 10.6, 11.2]
        self.assertEqual(la.cap_line_span(times), times)

    def test_wild_span_is_compressed(self):
        times = [10.0, 13.3, 16.9, 20.5]
        capped = la.cap_line_span(times)
        self.assertEqual(capped[0], 10.0)
        self.assertAlmostEqual(capped[-1] - capped[0], la.line_span_cap(4))
        self.assertTrue(capped[0] < capped[1] < capped[2] < capped[3])

    def test_song_pace(self):
        self.assertIsNone(la.song_pace([]))
        self.assertIsNone(la.song_pace([[1.0, 2.0]]))          # too short to count
        self.assertIsNone(la.song_pace([[0.0, 1.0, 2.0], [10.0, 10.5, 11.0, 11.5]]))  # fewer than 3 lines
        self.assertAlmostEqual(
            la.song_pace([[0.0, 1.0, 2.0], [10.0, 10.5, 11.0, 11.5], [20.0, 20.8, 21.6], [None, None, 5.0]]),
            0.8)  # median of 1.0, 0.5, 0.8; the poor line is ignored

    def test_ballad_pace_loosens_cap(self):
        # A ballad singing a word per second keeps its long line...
        times = [10.0 + k for k in range(10)]
        self.assertEqual(la.cap_line_span(times, pace=0.95), times)
        # ...while the same spread in a fast song is compressed.
        capped = la.cap_line_span(times, pace=0.3)
        self.assertLess(capped[-1] - capped[0], 8.0)
        self.assertEqual(capped[0], 10.0)

    def test_huge_single_gap_is_closed(self):
        # "...and hooooow" dragged into the instrumental break after it.
        times = [10.0, 10.4, 10.8, 11.2, 19.0]
        capped = la.cap_line_span(times, pace=0.5)
        self.assertEqual(capped[:4], times[:4])
        self.assertAlmostEqual(capped[4], 11.2 + max(la.GAP_FLOOR, la.GAP_PACE_FACTOR * 0.5))
        # A ballad's real 3.5 s mid-line pause survives.
        times = [10.0, 11.0, 14.5, 15.5]
        self.assertEqual(la.cap_line_span(times, pace=0.9), times)

    def test_convert_applies_cap(self):
        lines = la.parse_lrc("[00:10.00] had a bad day\n")
        out, _ = la.convert_lines(lines, lambda s, e, t: [("had", 10.0, 10.2, .9), ("a", 13.0, 13.2, .9),
                                                          ("bad", 16.7, 17.0, .9), ("day", 20.3, 21.0, .9)])
        times = [la.parse_timestamp(*m.groups()) for m in la.WORD_TS_RE.finditer(out[0])]
        self.assertLessEqual(times[-1] - times[0], la.line_span_cap(4) + 0.01)

    def test_word_level_neighbour_is_stripped(self):
        lines = la.parse_lrc(WORD_FILE + "[00:40.00] plain line\n")
        ctx = la.build_context(lines, 2, 70.0, 200.0)
        self.assertEqual(ctx.text, "That tonight's gonna be\nplain line")
        self.assertEqual(ctx.first_token, 4)

    def test_slice_never_exceeds_thirty_seconds(self):
        lines = la.parse_lrc("[00:10.00] a\n[00:17.00] b\n[00:24.00] c\n[00:44.00] d\n")
        ctx = la.build_context(lines, 1, 24.0, 200.0)
        self.assertLessEqual(ctx.slice_end - ctx.slice_start, la.MAX_LINE_WINDOW)
        self.assertTrue(ctx.has_prev)


class RetimeTests(unittest.TestCase):
    TEXT = "[00:10.00] one two three\n[00:14.00] four five\n[00:17.00] six seven\n"

    def convert(self, offset, **kw):
        lines = la.parse_lrc(self.TEXT)
        return la.convert_lines(lines, context_align(lines, offset), **kw)

    def test_retimed_line_start_bounds(self):
        self.assertEqual(la.retimed_line_start(10.3, 10.0, 14.0), 10.3)
        self.assertEqual(la.retimed_line_start(13.0, 10.0, 14.0), 10.0 + la.MAX_LATE_SHIFT)
        self.assertEqual(la.retimed_line_start(None, 10.0, 14.0), 10.0)
        # Small differences are left alone: the original tag is probably right.
        self.assertEqual(la.retimed_line_start(10.1, 10.0, 14.0), 10.0)
        # Tags are never moved earlier: an early estimate is aligner noise.
        self.assertEqual(la.retimed_line_start(9.8, 10.0, 14.0), 10.0)
        self.assertEqual(la.retimed_line_start(9.0, 10.0, 14.0), 10.0)
        # Never reaches the next line even when the aligner says so.
        self.assertAlmostEqual(la.retimed_line_start(10.9, 10.0, 10.5), 10.5 - la.MAX_EARLY_SHIFT - la.MIN_WORD_STEP)

    def test_decide_retiming(self):
        B = la.FIRST_WORD_BIAS
        self.assertEqual(la.decide_retiming([]), ("none", 0.0))
        # Accurate file: nothing to do.
        self.assertEqual(la.decide_retiming([(10.0, 10.0 + B), (14.0, 14.05 + B), (17.0, 16.9 + B)])[0], "none")
        # Early file with uneven per-line error: per-line later-only.
        mode, amount = la.decide_retiming([(10.0, 10.5 + B), (14.0, 14.2 + B), (17.0, 17.9 + B)])
        self.assertEqual(mode, "per_line")
        # A late-heard minority in an accurate file does not open the gate.
        self.assertEqual(la.decide_retiming([(10.0, 10.0 + B), (14.0, 14.8 + B), (17.0, 17.0 + B)])[0], "none")
        # Consistent offset across many lines: whole-file shift (either direction).
        est = [(10.0 * k, 10.0 * k + 0.9 + B) for k in range(1, 9)]
        mode, amount = la.decide_retiming(est)
        self.assertEqual(mode, "global")
        self.assertAlmostEqual(amount, 0.9)
        est = [(10.0 * k, 10.0 * k - 0.6 + B) for k in range(1, 9)]
        self.assertEqual(la.decide_retiming(est)[0], "global")
        # Too few lines for a whole-file decision falls back to per-line.
        est = [(10.0 * k, 10.0 * k + 0.9 + B) for k in range(1, 4)]
        self.assertEqual(la.decide_retiming(est)[0], "per_line")

    def test_shift_line(self):
        ln = la.parse_lrc("[00:10.00] one two\n[00:12.00]\n[00:28.90]<00:28.90>I <00:29.34>got\n[ar:x]\n")
        s = [la.shift_line(l, 0.9, 2) for l in ln]
        self.assertEqual(s[0].raw, "[00:10.90] one two")
        self.assertAlmostEqual(s[0].start, 10.9)
        self.assertEqual(s[1].raw, "[00:12.90]")
        self.assertEqual(s[2].raw, "[00:29.80]<00:29.80>I <00:30.24>got")
        self.assertEqual(s[3].raw, "[ar:x]")
        self.assertEqual(la.shift_line(ln[0], -20.0, 2).raw, "[00:00.00] one two")

    def test_whole_file_offset_is_applied(self):
        text = "".join(f"[00:{10 + 4 * k:02d}.00] word{k} more\n" for k in range(8))
        lines = la.parse_lrc(text)
        out, stats = la.convert_lines(lines, context_align(lines, 0.9 + la.FIRST_WORD_BIAS))
        self.assertAlmostEqual(stats.global_offset, 0.9)
        self.assertTrue(out[0].startswith("[00:10.90]<00:10.90>word0"), out[0])
        self.assertTrue(out[7].startswith("[00:38.90]<00:38.90>word7"), out[7])
        # Keep-line-times switches all of this off.
        out, stats = la.convert_lines(lines, context_align(lines, 0.9 + la.FIRST_WORD_BIAS), retime_lines=False)
        self.assertEqual(stats.global_offset, 0.0)
        self.assertTrue(out[0].startswith("[00:10.00]"), out[0])

    def test_tag_moves_to_sung_first_word(self):
        # The aligner hears first words FIRST_WORD_BIAS late; that is corrected.
        out, stats = self.convert(0.3 + la.FIRST_WORD_BIAS)
        # The first line has no previous line as context, so its tag is kept.
        self.assertTrue(out[0].startswith("[00:10.00]<00:10.00>one <00:10.92>two"), out[0])
        self.assertTrue(out[1].startswith("[00:14.30]<00:14.30>four <00:14.92>five"), out[1])
        self.assertTrue(out[2].startswith("[00:17.30]<00:17.30>six"), out[2])
        self.assertEqual(stats.lines_retimed, 2)
        self.assertEqual(stats.lines_aligned, 3)

    def test_accurate_tags_are_left_alone(self):
        out, stats = self.convert(0.1 + la.FIRST_WORD_BIAS)
        self.assertTrue(out[1].startswith("[00:14.00]<00:14.00>four"), out[1])
        self.assertEqual(stats.lines_retimed, 0)

    def test_tags_are_never_moved_earlier(self):
        out, stats = self.convert(-0.2 + la.FIRST_WORD_BIAS)
        self.assertTrue(out[1].startswith("[00:14.00]<00:14.00>four"), out[1])
        self.assertEqual(stats.lines_retimed, 0)
        out, _ = self.convert(-0.45 + la.FIRST_WORD_BIAS)
        self.assertTrue(out[1].startswith("[00:14.00]<00:14.00>four"), out[1])

    def test_late_shift_is_capped(self):
        out, _ = self.convert(2.5)
        self.assertTrue(out[1].startswith("[00:14.35]<00:14.35>four"), out[1])

    def test_keep_line_times(self):
        out, stats = self.convert(0.3 + la.FIRST_WORD_BIAS, retime_lines=False)
        self.assertTrue(out[1].startswith("[00:14.00]<00:14.00>four"), out[1])
        self.assertEqual(stats.lines_retimed, 0)

    def test_fallback_lines_keep_their_tag(self):
        out, stats = la.convert_lines(la.parse_lrc(self.TEXT), lambda s, e, t: [])
        self.assertTrue(out[1].startswith("[00:14.00]<00:14.00>four"), out[1])
        self.assertEqual(stats.lines_retimed, 0)

    def test_words_never_pass_the_next_retimed_tag(self):
        # Line 1's words run late; line 2's tag moves 0.3s later. Every word of
        # line 1 must still sit before line 2's new tag.
        when = {"one": 10.0, "two": 13.9, "three": 14.4, "four": 14.3 + la.FIRST_WORD_BIAS, "five": 14.8,
                "six": 17.3 + la.FIRST_WORD_BIAS, "seven": 17.7}

        def fn(start, end, text):
            return [(w, when[w], when[w] + 0.2, .9) for w in text.split()]
        out, _ = la.convert_lines(la.parse_lrc(self.TEXT), fn)
        self.assertTrue(out[1].startswith("[00:14.30]"), out[1])
        times = [la.parse_timestamp(*m.groups()) for m in la.WORD_TS_RE.finditer(out[0])]
        self.assertEqual(len(times), 3)
        self.assertTrue(all(t < 14.3 for t in times), times)
        self.assertTrue(times[0] < times[1] < times[2], times)

    def test_early_next_tag_makes_room_for_previous_tail(self):
        # Line A's last word is sung at 14.02, line B is tagged 14.00 but its
        # first word is heard at 14.40: B's tag moves to just after A's tail.
        B = la.FIRST_WORD_BIAS
        when = {"one": 10.0, "two": 12.0, "three": 14.02, "four": 14.40 + B, "five": 14.9,
                "six": 17.0 + B, "seven": 17.4}

        def fn(start, end, text):
            return [(w, when[w], when[w] + 0.2, .9) for w in text.split()]
        out, stats = la.convert_lines(la.parse_lrc(self.TEXT), fn)
        tag_b = la.parse_timestamp(*la.LINE_TS_RE.match(out[1]).groups())
        self.assertGreaterEqual(tag_b, 14.17, out[1])
        self.assertLessEqual(tag_b, 14.40, out[1])
        times = [la.parse_timestamp(*m.groups()) for m in la.WORD_TS_RE.finditer(out[0])]
        self.assertGreater(tag_b - times[-1], 0.1)       # last word now has room to show
        self.assertGreaterEqual(stats.lines_retimed, 1)

    def test_true_overlap_does_not_move_tag(self):
        # Line B really starts before A's last word: nothing to gain by moving B.
        when = {"one": 10.0, "two": 12.0, "three": 14.3, "four": 14.1, "five": 14.6,
                "six": 17.0, "seven": 17.4}

        def fn(start, end, text):
            return [(w, when[w], when[w] + 0.2, .9) for w in text.split()]
        out, _ = la.convert_lines(la.parse_lrc(self.TEXT), fn)
        self.assertTrue(out[1].startswith("[00:14.00]"), out[1])

    def test_tail_room_is_capped(self):
        raw = {0: [10.0, 15.5], 1: [16.9, 17.2], 2: [30.0, 30.4]}
        lines = la.parse_lrc("[00:10.00] a b\n[00:14.00] c d\n[00:30.00] e f\n")
        eff = {0: 10.0, 1: 14.0, 2: 30.0}
        self.assertEqual(la.make_room_for_tails(lines, raw, eff), 1)
        self.assertAlmostEqual(eff[1], 14.0 + la.MAX_TAIL_SHIFT)

    def test_tags_stay_in_order_for_dense_lines(self):
        lines = la.parse_lrc("[00:10.00] a b\n[00:10.40] c d\n[00:10.80] e f\n")
        out, _ = la.convert_lines(lines, context_align(lines, 0.9))
        starts = [la.parse_timestamp(*la.LINE_TS_RE.match(l).groups()) for l in out]
        self.assertTrue(starts[0] < starts[1] < starts[2], starts)


class FileMatchingTests(unittest.TestCase):
    def test_normalize_stem(self):
        self.assertEqual(la.normalize_stem("Gangsta#U2019s Paradise"), "gangsta’s paradise")
        self.assertEqual(la.normalize_stem("  Two   Spaces "), "two spaces")

    def test_match_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "music").mkdir()
            (root / "lyrics").mkdir()
            (root / "music" / "Gangsta’s Paradise.mp3").write_bytes(b"")
            (root / "music" / "Some Song.flac").write_bytes(b"")
            (root / "music" / "Next To.mp3").write_bytes(b"")
            (root / "lyrics" / "Gangsta#U2019s Paradise.lrc").write_text("x")
            (root / "lyrics" / "some song.lrc").write_text("x")
            (root / "lyrics" / "Missing.lrc").write_text("x")
            (root / "music" / "Next To.lrc").write_text("x")
            index = la.index_audio(root / "music", recursive=False)
            self.assertEqual(
                la.match_audio(root / "lyrics" / "Gangsta#U2019s Paradise.lrc", index, root / "lyrics", root / "music"),
                root / "music" / "Gangsta’s Paradise.mp3")
            self.assertEqual(
                la.match_audio(root / "lyrics" / "some song.lrc", index, root / "lyrics", root / "music").name,
                "Some Song.flac")
            self.assertIsNone(la.match_audio(root / "lyrics" / "Missing.lrc", index, root / "lyrics", root / "music"))
            self.assertEqual(
                la.match_audio(root / "music" / "Next To.lrc", {}, root / "music", root / "music").name, "Next To.mp3")

    def test_output_path(self):
        lyrics = Path("/l")
        out = Path("/o")
        self.assertEqual(la.output_path_for(Path("/l/sub/a.lrc"), lyrics, out, True), Path("/o/sub/a.lrc"))
        self.assertEqual(la.output_path_for(Path("/l/sub/a.lrc"), lyrics, out, False), Path("/o/a.lrc"))


class FakeAligner:
    """Stands in for WhisperLineAligner in process_library tests."""
    device = "cpu"

    def __init__(self):
        self.loaded = []

    def load_file(self, path):
        self.loaded.append(path)
        return 200.0

    def detect_language(self, start=0.0):
        return "en"

    def make_align_fn(self, language):
        return fake_align()


class ProvenanceTests(unittest.TestCase):
    """Every converted file carries a [re:lrc-align <version> align word] tag."""

    def convert(self, text, tmp):
        music = Path(tmp)
        (music / "Song.mp3").write_bytes(b"")
        (music / "Song.lrc").write_text(text, encoding="utf-8")
        out = music / "out" / "Song.lrc"
        result = la.convert_file(music / "Song.lrc", music / "Song.mp3", FakeAligner(), out)
        self.assertEqual(result.status, "converted")
        return out.read_text(encoding="utf-8")

    @staticmethod
    def tags(content):
        return [m.group(1).split() for line in content.splitlines() if (m := la.PROVENANCE_RE.match(line))]

    def test_converted_file_has_exactly_one_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = self.convert(LINE_FILE, tmp)
        tags = self.tags(content)
        self.assertEqual(tags, [["lrc-align", la.__version__, "align", "word"]])
        self.assertTrue(content.startswith("[re:lrc-align "), content[:60])
        # The tag comes before the first timestamped line.
        first_ts = next(i for i, l in enumerate(content.splitlines()) if la.LINE_TS_RE.match(l))
        self.assertLess(0, first_ts)

    def test_existing_tag_is_replaced_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = self.convert("[re:beetdrop 0.62.0 apple line]\n" + LINE_FILE, tmp)
        self.assertNotIn("beetdrop", content)
        self.assertEqual(self.tags(content), [["lrc-align", la.__version__, "align", "word", "from-apple"]])

    def test_input_without_tag_gets_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = self.convert(LINE_FILE, tmp)
        self.assertEqual(len(self.tags(content)), 1)

    def test_tagging_changes_nothing_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = self.convert("[re:beetdrop 0.62.0 apple line]\n" + LINE_FILE, tmp)
        lines = la.parse_lrc("[re:beetdrop 0.62.0 apple line]\n" + LINE_FILE)
        untagged, _ = la.convert_lines(lines, FakeAligner().make_align_fn("en"), audio_duration=200.0,
                                       decimals=la.detect_decimals(lines))
        untagged = [l for l in untagged if not la.PROVENANCE_RE.match(l)]
        stripped = [l for l in content.splitlines() if not la.PROVENANCE_RE.match(l)]
        self.assertEqual(stripped, untagged)

    def test_skipped_word_level_file_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp)
            (music / "Word.mp3").write_bytes(b"")
            original = ("[re:beetdrop 0.62.0 apple word]\n" + WORD_FILE).encode("utf-8")
            (music / "Word.lrc").write_bytes(original)
            summary = la.process_library(music, music, music, FakeAligner(), log=lambda m: None)
            self.assertEqual(summary.count("skipped_word_level"), 1)
            self.assertEqual((music / "Word.lrc").read_bytes(), original)
            self.assertFalse((music / "Word.lrc.bak").exists())

    def test_converting_twice_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp)
            (music / "Song.mp3").write_bytes(b"")
            (music / "Song.lrc").write_text(LINE_FILE)
            la.process_library(music, music, music, FakeAligner(), log=lambda m: None)
            first = (music / "Song.lrc").read_bytes()
            self.assertEqual(len(self.tags(first.decode())), 1)
            summary = la.process_library(music, music, music, FakeAligner(), log=lambda m: None)
            self.assertEqual(summary.count("converted"), 0)
            self.assertEqual((music / "Song.lrc").read_bytes(), first)

    def test_timing_field_describes_the_file(self):
        self.assertEqual(la.stamp_provenance(["[ar:x]", "[00:10.00] plain"]),
                         ["[re:lrc-align %s align line]" % la.__version__, "[ar:x]", "[00:10.00] plain"])
        self.assertEqual(la.stamp_provenance(["[re:a b c d]", "[re:second]", "[00:10.00]<00:10.00>w"]),
                         ["[re:lrc-align %s align word from-c]" % la.__version__, "[00:10.00]<00:10.00>w"])


class ProcessLibraryTests(unittest.TestCase):
    def test_process_library_skips_and_converts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            music, out = root / "music", root / "out"
            music.mkdir()
            (music / "Line.mp3").write_bytes(b"")
            (music / "Line.lrc").write_text(LINE_FILE)
            (music / "Word.mp3").write_bytes(b"")
            (music / "Word.lrc").write_text(WORD_FILE)
            (music / "NoAudio.lrc").write_text(LINE_FILE)
            (music / "Empty.mp3").write_bytes(b"")
            (music / "Empty.lrc").write_text("[ar:x]\n")
            logs = []
            aligner = FakeAligner()
            summary = la.process_library(music, music, out, aligner, log=logs.append)
            self.assertEqual(summary.count("converted"), 1)
            self.assertEqual(summary.count("skipped_word_level"), 1)
            self.assertEqual(summary.count("skipped_no_audio"), 1)
            self.assertEqual(summary.count("skipped_no_lyrics"), 1)
            self.assertEqual(aligner.loaded, [music / "Line.mp3"])
            converted = (out / "Line.lrc").read_text()
            self.assertIn("[00:15.23]<00:15.23>Oh,", converted)
            self.assertTrue(la.is_word_level(la.parse_lrc(converted)))
            # A report lists the questionable files first.
            report = (out / la.REPORT_NAME).read_text()
            self.assertIn("FLAGGED (1)", report)
            self.assertIn("NoAudio.lrc", report.split("ALL FILES:")[0])
            self.assertIn("no matching audio file", report)
            self.assertIn("converted            Line.lrc", report)
            # Running again skips the already converted output.
            summary2 = la.process_library(music, music, out, aligner, log=logs.append)
            self.assertEqual(summary2.count("converted"), 0)
            self.assertEqual(summary2.count("skipped_word_level"), 2)

    def test_reconvert_redoes_only_this_tools_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp)
            for name in ("Mine", "MineNoBak", "Theirs", "Plain"):
                (music / f"{name}.mp3").write_bytes(b"")
            ours = "[re:lrc-align 0.9.0 align word]\n" + WORD_FILE
            (music / "Mine.lrc").write_text(ours)
            (music / "Mine.lrc.bak").write_text("[00:28.90] I got a feeling\n[00:36.66] That tonight's gonna be\n")
            (music / "MineNoBak.lrc").write_text(ours)
            (music / "Theirs.lrc").write_text("[re:beetdrop 0.62.0 apple word]\n" + WORD_FILE)
            (music / "Plain.lrc").write_text(LINE_FILE)
            theirs_before = (music / "Theirs.lrc").read_bytes()
            summary = la.process_library(music, music, music, FakeAligner(), log=lambda m: None, reconvert=True)
            self.assertEqual(summary.count("converted"), 3)
            self.assertEqual(summary.count("skipped_word_level"), 1)
            self.assertEqual((music / "Theirs.lrc").read_bytes(), theirs_before)
            mine = (music / "Mine.lrc").read_text()
            self.assertTrue(mine.startswith(f"[re:lrc-align {la.__version__} align word]"), mine[:50])
            self.assertIn("[00:28.90]<00:28.90>I ", mine)                # re-aligned from the .bak
            self.assertNotEqual(mine, ours)
            # The original backup is kept, not overwritten with the word-level file.
            self.assertEqual((music / "Mine.lrc.bak").read_text().splitlines()[0], "[00:28.90] I got a feeling")
            nobak = (music / "MineNoBak.lrc").read_text()
            self.assertIn("[00:28.90]<00:28.90>I ", nobak)               # re-aligned from stripped tags
            self.assertNotEqual(nobak, ours)
            self.assertEqual(len([l for l in nobak.splitlines() if l.startswith("[re:")]), 1)
            # Without the flag nothing of ours is touched.
            before = (music / "Mine.lrc").read_bytes()
            summary = la.process_library(music, music, music, FakeAligner(), log=lambda m: None)
            self.assertEqual(summary.count("converted"), 0)
            self.assertEqual((music / "Mine.lrc").read_bytes(), before)

    def test_line_level_source(self):
        src = la.line_level_source("[re:lrc-align 1.0.0 align word]\n[ar:x]\n" + WORD_FILE + "[00:40.00]\n")
        self.assertEqual(src, "[ar:x]\n[00:28.90] I got a feeling\n[00:36.66] That tonight's gonna be\n[00:40.00]\n")

    def test_flags(self):
        ok = la.FileResult(Path("a.lrc"), "converted", stats=la.ConvertStats(lines_aligned=10))
        self.assertEqual(ok.flags(), [])
        shifted = la.FileResult(Path("a.lrc"), "converted", stats=la.ConvertStats(lines_aligned=10, global_offset=0.9))
        self.assertIn("+0.90 s", shifted.flags()[0])
        poor = la.FileResult(Path("a.lrc"), "converted", stats=la.ConvertStats(lines_aligned=5, lines_fallback=5))
        self.assertIn("5 of 10 lines", poor.flags()[0])
        self.assertIn("failed: boom", la.FileResult(Path("a.lrc"), "failed", message="boom").flags()[0])
        self.assertEqual(la.FileResult(Path("a.lrc"), "skipped_word_level").flags(), [])
        rough = la.FileResult(Path("a.lrc"), "converted",
                              stats=la.ConvertStats(lines_aligned=10, line_confidences=[0.3, 0.4]))
        self.assertIn("low alignment confidence (0.35)", rough.flags()[0])
        fine = la.FileResult(Path("a.lrc"), "converted",
                             stats=la.ConvertStats(lines_aligned=10, line_confidences=[0.8, 0.9]))
        self.assertEqual(fine.flags(), [])

    def test_confidence_is_collected(self):
        lines = la.parse_lrc("[00:10.00] one two\n[00:14.00] three\n")
        _, stats = la.convert_lines(lines, context_align(lines))
        self.assertAlmostEqual(stats.confidence, 0.9)

    def test_in_place_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp)
            (music / "Line.mp3").write_bytes(b"")
            (music / "Line.lrc").write_text(LINE_FILE)
            summary = la.process_library(music, music, music, FakeAligner(), log=lambda m: None)
            self.assertEqual(summary.count("converted"), 1)
            self.assertEqual((music / "Line.lrc.bak").read_text(), LINE_FILE)
            self.assertTrue(la.is_word_level(la.parse_lrc((music / "Line.lrc").read_text())))

    def test_backup_survives_mount_without_metadata_support(self):
        import shutil as _shutil
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "a.lrc", Path(tmp) / "a.lrc.bak"
            src.write_text("x")
            real = _shutil.copy2

            def refuse(*a, **k):
                raise OSError(95, "Operation not supported")
            _shutil.copy2 = refuse
            try:
                la.backup_file(src, dst)
            finally:
                _shutil.copy2 = real
            self.assertEqual(dst.read_text(), "x")

    def test_stop_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp)
            for name in ("A", "B"):
                (music / f"{name}.mp3").write_bytes(b"")
                (music / f"{name}.lrc").write_text(LINE_FILE)
            summary = la.process_library(music, music, music / "out", FakeAligner(),
                                         log=lambda m: None, should_stop=lambda: True)
            self.assertEqual(len(summary.results), 0)


if __name__ == "__main__":
    unittest.main()
