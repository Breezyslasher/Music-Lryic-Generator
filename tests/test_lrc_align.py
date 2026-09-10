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
        self.assertAlmostEqual(ctx.slice_end, 20.0)  # audio ends

    def test_far_neighbours_are_not_context(self):
        lines = la.parse_lrc("[00:10.00] one\n[00:30.00] two\n[00:50.00] three\n")
        ctx = la.build_context(lines, 1, 50.0, 200.0)
        self.assertEqual(ctx.text, "two")
        self.assertFalse(ctx.has_prev)
        self.assertAlmostEqual(ctx.slice_end, 30.0 + la.SOLO_TAIL_MAX)

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
        self.assertEqual(la.retimed_line_start(9.8, 10.0, 14.0), 9.8)
        self.assertEqual(la.retimed_line_start(9.0, 10.0, 14.0), 10.0 - la.MAX_EARLY_SHIFT)
        self.assertEqual(la.retimed_line_start(13.0, 10.0, 14.0), 10.0 + la.MAX_LATE_SHIFT)
        self.assertEqual(la.retimed_line_start(None, 10.0, 14.0), 10.0)
        # Never reaches the next line even when the aligner says so.
        self.assertAlmostEqual(la.retimed_line_start(10.9, 10.0, 10.5), 10.5 - la.MAX_EARLY_SHIFT - la.MIN_WORD_STEP)

    def test_tag_moves_to_sung_first_word(self):
        out, stats = self.convert(0.3)
        # The first line has no previous line as context, so its tag is kept.
        self.assertTrue(out[0].startswith("[00:10.00]<00:10.00>one <00:10.80>two"), out[0])
        self.assertTrue(out[1].startswith("[00:14.30]<00:14.30>four <00:14.80>five"), out[1])
        self.assertTrue(out[2].startswith("[00:17.30]<00:17.30>six"), out[2])
        self.assertEqual(stats.lines_retimed, 2)
        self.assertEqual(stats.lines_aligned, 3)

    def test_tag_moves_earlier_within_bound(self):
        out, _ = self.convert(-0.2)
        self.assertTrue(out[1].startswith("[00:13.80]<00:13.80>four"), out[1])
        out, _ = self.convert(-0.45)
        self.assertTrue(out[1].startswith("[00:13.70]<00:13.70>four"), out[1])

    def test_late_shift_is_capped(self):
        out, _ = self.convert(2.5)
        self.assertTrue(out[1].startswith("[00:14.50]<00:14.50>four"), out[1])

    def test_keep_line_times(self):
        out, stats = self.convert(0.3, retime_lines=False)
        self.assertTrue(out[1].startswith("[00:14.00]<00:14.00>four"), out[1])
        self.assertEqual(stats.lines_retimed, 0)

    def test_fallback_lines_keep_their_tag(self):
        out, stats = la.convert_lines(la.parse_lrc(self.TEXT), lambda s, e, t: [])
        self.assertTrue(out[1].startswith("[00:14.00]<00:14.00>four"), out[1])
        self.assertEqual(stats.lines_retimed, 0)

    def test_words_never_pass_the_next_retimed_tag(self):
        # Line 1's words run late; line 2's tag moves 0.3s later. Every word of
        # line 1 must still sit before line 2's new tag.
        when = {"one": 10.0, "two": 13.9, "three": 14.4, "four": 14.3, "five": 14.8, "six": 17.0, "seven": 17.4}

        def fn(start, end, text):
            return [(w, when[w], when[w] + 0.2, .9) for w in text.split()]
        out, _ = la.convert_lines(la.parse_lrc(self.TEXT), fn)
        self.assertTrue(out[1].startswith("[00:14.30]"), out[1])
        times = [la.parse_timestamp(*m.groups()) for m in la.WORD_TS_RE.finditer(out[0])]
        self.assertEqual(len(times), 3)
        self.assertTrue(all(t < 14.3 for t in times), times)
        self.assertTrue(times[0] < times[1] < times[2], times)

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
            # Running again skips the already converted output.
            summary2 = la.process_library(music, music, out, aligner, log=logs.append)
            self.assertEqual(summary2.count("converted"), 0)
            self.assertEqual(summary2.count("skipped_word_level"), 2)

    def test_in_place_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp)
            (music / "Line.mp3").write_bytes(b"")
            (music / "Line.lrc").write_text(LINE_FILE)
            summary = la.process_library(music, music, music, FakeAligner(), log=lambda m: None)
            self.assertEqual(summary.count("converted"), 1)
            self.assertEqual((music / "Line.lrc.bak").read_text(), LINE_FILE)
            self.assertTrue(la.is_word_level(la.parse_lrc((music / "Line.lrc").read_text())))

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
