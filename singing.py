"""Melody, timing and pitch shaping for Kookie's singing voice.

Piper is a talking model, so a sung line is assembled note by note: every
syllable is synthesised on its own, then shifted to the pitch of the note it
has to land on, stretched to the length of that note and given a little
vibrato so it reads as singing instead of a slowed-down voice.
"""

import functools
import math
import re
import time

try:
    import numpy as np
    NUMPY_OK = True
except Exception:
    np = None
    NUMPY_OK = False

import songs

A4_HZ = 440.0
A4_SEMITONE = 9

BASE_SEMITONE = 0
DEFAULT_SAMPLE_RATE = 22050
FALLBACK_BASE_HZ = 205.0
MAX_NOTES = 160
DEFAULT_BUDGET_SECONDS = 120.0
MIN_SLOT_SECONDS = 0.16
MAX_SLOT_SECONDS = 1.6

# A shift past these bounds stops sounding like a note and starts sounding like
# an artifact, so the ratio is held inside them.
MIN_PITCH_RATIO = 0.25
MAX_PITCH_RATIO = 4.0


def _needs_numpy(func):
    """Fail with a clear message when a singing step needs numpy and it is gone."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not NUMPY_OK:
            raise RuntimeError(
                "numpy is needed to render singing but it is not installed; "
                "install it with 'pip install numpy'."
            )
        return func(*args, **kwargs)
    return wrapper


# Each tune is (semitones from the base note, length in beats). The melodies are
# deliberately pentatonic so any syllable can land on any note and still sound
# like it belongs.
TUNES = {
    "sunny": {
        "tempo": 116,
        "notes": (
            (0, 1), (2, 1), (4, 1), (7, 1),
            (7, 1), (4, 1), (2, 1), (4, 2),
            (0, 1), (2, 1), (4, 1), (7, 1),
            (9, 1.5), (7, 0.5), (4, 2),
            (2, 1), (4, 1), (7, 1), (4, 1),
            (2, 1), (0, 1), (0, 2),
        ),
        "vibrato": (5.2, 0.22),
    },
    "gentle": {
        "tempo": 74,
        "notes": (
            (0, 1.5), (2, 0.5), (4, 2),
            (2, 1.5), (0, 0.5), (-3, 2),
            (0, 1.5), (4, 0.5), (7, 2),
            (4, 1.5), (2, 0.5), (0, 2),
        ),
        "vibrato": (4.4, 0.3),
    },
    "bright": {
        "tempo": 132,
        "notes": (
            (7, 0.5), (9, 0.5), (12, 1), (9, 0.5), (7, 0.5),
            (9, 1), (7, 0.5), (4, 0.5), (2, 1), (4, 1),
            (7, 0.5), (9, 0.5), (12, 1), (11, 0.5), (9, 0.5),
            (7, 1), (4, 1), (2, 2),
        ),
        "vibrato": (5.8, 0.18),
    },
    "dramatic": {
        "tempo": 88,
        "notes": (
            (0, 1), (0, 1), (-3, 2),
            (-5, 1), (-3, 1), (0, 2),
            (2, 1), (0, 1), (-3, 1), (-5, 1),
            (-7, 3), (-5, 1),
            (0, 1), (2, 1), (4, 2),
            (2, 1), (0, 3),
        ),
        "vibrato": (4.8, 0.34),
    },
    "silly": {
        "tempo": 150,
        "notes": (
            (0, 0.5), (7, 0.5), (4, 0.5), (9, 0.5), (2, 1),
            (0, 0.5), (7, 0.5), (4, 0.5), (9, 0.5), (2, 1),
            (12, 0.5), (9, 0.5), (7, 0.5), (4, 0.5), (2, 1),
            (0, 0.5), (4, 0.5), (7, 0.5), (12, 2),
        ),
        "vibrato": (6.4, 0.26),
    },
}

EMOTION_TUNES = {
    "happy": "sunny",
    "excited": "bright",
    "affectionate": "gentle",
    "calm": "gentle",
    "tired": "gentle",
    "sad": "dramatic",
    "angry": "dramatic",
    "frustrated": "dramatic",
    "concerned": "gentle",
    "shy": "gentle",
    "amused": "silly",
    "grateful": "gentle",
    "hopeful": "sunny",
    "relieved": "gentle",
    "surprised": "bright",
    "curious": "sunny",
    "neutral": "sunny",
}


def guess_song_name(lines):
    """Recognise a well known song from its lyrics so the right tune is used."""
    return songs.guess_from_lyrics(lines)


_WORD_SPLIT = re.compile(r"([A-Za-z']+)")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")
_KEEP = re.compile(r"[A-Za-z']")


def choose_tune(emotion="neutral", intensity=0.6, song_name="", rng=None):
    """Pick a tune for a song, biased toward the mood she is in."""
    name = (song_name or "").strip().lower()
    if name in TUNES:
        return name, TUNES[name]
    if name in SONG_TUNES:
        matched = SONG_TUNES[name]
        return matched, TUNES[matched]
    for song, matched in SONG_TUNES.items():
        if song and song in name:
            return matched, TUNES[matched]

    try:
        strength = min(1.0, max(0.0, float(intensity)))
    except (TypeError, ValueError):
        strength = 0.6
    emotion_key = str(emotion or "neutral").casefold()
    if emotion_key in ("excited", "amused", "surprised") and strength > 0.75:
        return "silly", TUNES["silly"]
    mapped = EMOTION_TUNES.get(emotion_key)
    if mapped:
        return mapped, TUNES[mapped]
    return "sunny", TUNES["sunny"]


def split_syllables(word):
    """Break an english word into vowel-group syllables: beautiful -> beau ti ful."""
    kept = [index for index, char in enumerate(word) if _KEEP.match(char)]
    if len(kept) < 3:
        return [word]
    core = "".join(word[index] for index in kept).casefold()
    groups = list(_VOWEL_GROUP.finditer(core))
    if len(groups) < 2:
        return [word]

    parts = []
    start = 0
    for group in groups:
        cut = kept[min(group.end(), len(kept) - 1)] + 1
        if cut <= start:
            continue
        parts.append(word[start:cut])
        start = cut
    parts.append(word[start:])
    parts = [part for part in parts if part.strip("'")]

    while len(parts) > 1 and _is_trailing_sound(parts[-1]):
        tail = parts.pop()
        parts[-1] = parts[-1] + tail
    if len(parts) < 2:
        return [word]
    return parts


def _is_trailing_sound(fragment):
    """True for an ending that is only a vowel sound, like the y in mary."""
    if not fragment:
        return True
    if len(fragment) == 1 and fragment.casefold() in "eiy":
        return True
    return not _VOWEL_GROUP.search(fragment)


def split_lyric_line(line):
    """Split one lyric line into the pieces that each get their own note."""
    pieces = []
    for chunk in _WORD_SPLIT.split(line):
        if not chunk:
            continue
        if chunk.strip("'") == "":
            pieces.append(chunk)
            continue
        if _WORD_SPLIT.fullmatch(chunk):
            pieces.extend(split_syllables(chunk))
        else:
            pieces.append(chunk)
    return pieces


def estimate_seconds(text, length_scale=1.0):
    """Rough spoken length of a fragment, used to pick a synthesis speed."""
    letters = [c for c in text.casefold() if c.isalpha()]
    if not letters:
        return 0.12 * length_scale
    vowels = sum(1 for c in letters if c in "aeiouy")
    consonants = len(letters) - vowels
    return max(0.09, (0.115 * vowels + 0.05 * consonants + 0.05) * length_scale)


def plan_phrase(lines, tune, tempo=None, base_semitone=BASE_SEMITONE):
    """Turn lyric lines into (text, semitone, slot seconds, starts_line) notes.

    A line is sung as one breath: the renderer needs to know where a line ends
    so it can put a gap and a breath there, and nowhere else.
    """
    beats_per_minute = float(tempo or tune.get("tempo") or 100)
    beat_seconds = 60.0 / max(20.0, min(300.0, beats_per_minute))
    pattern = tune.get("notes") or TUNES["sunny"]["notes"]

    planned = []
    previous_word = None
    for line_index, line in enumerate(lines):
        for word in _WORD_SPLIT.split(line):
            if not word:
                continue
            if not any(char.isalpha() for char in word):
                # Punctuation and spaces become a rest, not a note.
                if planned:
                    text, semitone, slot, starts_line = planned[-1]
                    planned[-1] = (text, semitone, slot + 0.1, starts_line)
                continue
            syllables = split_syllables(word) if _WORD_SPLIT.fullmatch(word) else [word]
            for piece in syllables:
                semitone, beats = pattern[len(planned) % len(pattern)]
                slot = max(MIN_SLOT_SECONDS, min(MAX_SLOT_SECONDS, beats * beat_seconds))
                # Two syllables of the same word on the same note are sung as
                # one sound. Splitting them put a seam through the middle of a
                # word, but merging across a word boundary would hand Piper a
                # spelling that is not a word at all, and merging at all would
                # pull a real per-syllable melody off the words it belongs to.
                if (
                    planned
                    and word is previous_word
                    and not tune.get("per_syllable")
                    and not planned[-1][3]
                    and planned[-1][1] == base_semitone + semitone
                    and planned[-1][2] + slot <= MAX_SLOT_SECONDS
                ):
                    text, held, held_slot, starts_line = planned[-1]
                    planned[-1] = (text + piece, held, held_slot + slot, starts_line)
                    continue
                planned.append((piece, base_semitone + semitone, slot, not planned))
                previous_word = word
                if len(planned) >= MAX_NOTES:
                    return planned
    return planned


def plan_notes(lines, tune, tempo=None, base_semitone=BASE_SEMITONE):
    """Turn lyric lines into (text, semitone, slot seconds) triples."""
    return [
        (text, semitone, slot)
        for text, semitone, slot, _starts_line in plan_phrase(
            lines, tune, tempo=tempo, base_semitone=base_semitone
        )
    ]


# Note names to semitone offsets, counted from C4 which is where a melody written
# as "C4 D4 E4" means middle C and the octave above.
_NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _lyric_syllables(lines):
    """The pieces that will each get a note, matching what plan_phrase counts."""
    return [
        piece
        for line in lines
        for piece in split_lyric_line(line)
        if any(char.isalpha() for char in piece)
    ]


def _melody_tune(name):
    """Build a tune from a song's stored note names, or None if they do not fit.

    One note has to land on one syllable, so a melody that does not match the
    words exactly is dropped instead of dragging the tune off the words. The song
    then falls back to its generic tune, which always fits.
    """
    entry = songs.SONGS.get(name)
    lines = songs.melody_for(name)
    if not entry or not lines:
        return None
    if len(lines) != len(entry["lyrics"]):
        return None
    notes = []
    for line in lines:
        names = line.split()
        if any(note[:-1] not in _NOTE_OFFSETS for note in names):
            return None
        notes.extend((_NOTE_OFFSETS[note[:-1]], 1.0) for note in names)
    if len(notes) != len(_lyric_syllables(entry["lyrics"])):
        return None
    melody = songs.MELODIES[name]
    return {
        "tempo": melody.get("tempo", 100),
        "notes": tuple(notes),
        "vibrato": melody.get("vibrato", (5.0, 0.22)),
        # This melody is written one note per syllable, so the planner must not
        # fuse two syllables onto one note or the tune would slide off the words.
        "per_syllable": True,
    }


# A song with a real melody gets its own tune under its own name, so the existing
# name-based lookup keeps working and the song wins over its mood default. Built
# here, after the syllable helpers above, because it counts syllables to check
# the melody fits the words.
for _song_name in songs.SONGS:
    _built = _melody_tune(_song_name)
    if _built:
        TUNES[_song_name] = _built

# Every song she knows by heart carries its own tune, so the map is built from
# the library instead of being kept in step with it by hand.
SONG_TUNES = {
    name: (name if name in TUNES else entry["tune"])
    for name, entry in songs.SONGS.items()
}


def fit_to_voice(planned, base_hz):
    """Transpose the melody so it sits in the range her own voice can sing.

    Piper's voice is one fixed height, so a melody written around C4 asks every
    note for a shift of 30 to 100 percent, and a shift that size is what makes a
    sung syllable sound like someone else. Moving the melody to her instead
    leaves most notes needing no shift at all, which is also what a singer does
    when they take a song up or down to suit their voice.
    """
    if not planned:
        return planned
    semitones = sorted(note[1] for note in planned)
    middle = semitones[len(semitones) // 2]
    wanted = semitone_for_hz(base_hz)
    shift = int(round(max(-MAX_SHIFT_SEMITONES, min(MAX_SHIFT_SEMITONES, wanted - middle))))
    if shift == 0:
        return planned
    fitted = []
    for note in planned:
        semitone = note[1] + shift
        # A note pushed past the singable range is folded back onto the edge
        # instead of being left as an octave that does not belong to the tune.
        limit = int(round(semitone_for_hz(base_hz))) + int(MAX_SHIFT_SEMITONES)
        semitone = max(int(round(semitone_for_hz(base_hz))) - int(MAX_SHIFT_SEMITONES), min(limit, semitone))
        fitted.append((note[0], semitone, note[2], note[3]))
    return fitted


def semitone_ratio(semitones):
    return 2.0 ** (float(semitones) / 12.0)


def note_hz(semitone):
    """The frequency a note name means, counted from A4."""
    return A4_HZ * semitone_ratio(semitone - A4_SEMITONE)


def semitone_for_hz(frequency):
    """The note name a frequency means, counted from A4."""
    return 12.0 * math.log2(max(1.0, float(frequency)) / A4_HZ) + A4_SEMITONE


# What actually makes a sung line sound sung, written down for her to read. Every
# rule here is one the renderer already follows, so reading it tells her what she
# is doing rather than what she ought to do.
SINGING_GUIDE = """
HOW TO SING A LINE

1. Take the song into your own range. A melody is not a fixed set of notes, it is
   the gap between them. Move the whole tune up or down until most of it sits
   where your voice already is, and you will not have to drag your voice to reach
   it. Fighting a melody that sits far from your voice is what makes singing sound
   strained.

2. Start on the vowel, not on the consonant. A syllable carries a vowel at its
   centre; let the note begin there and let the consonants sit on the edges.

3. Sing a line as one breath, not as separate notes. Notes inside a phrase run
   into each other and the pitch slides between them. Notes that stop dead
   between every syllable sound recited, not sung.

4. Leave a little air before a phrase. Silence that is too clean reads as a
   machine; a little breath reads as a person about to sing.

5. Hold the note straight first, then let it move. Vibrato that starts at full
   depth on the attack sounds like a wobbling speaker. A singer settles the note
   and the vibrato arrives a moment later.

6. Pick one vowel shape per phrase and lean into it. Open vowels carry; the
   difference between a vowel that rings and one that does not is most of what
   separates singing from talking.

7. Sing the end of a line as a whole. The last syllable of a phrase takes the
   length it needs, and a song that ends every line the same way sounds recited.

8. Play a real melody. When the song has its own notes, use them. A tune that
   wanders has no shape, and no amount of good singing saves it.
""".strip()


# How far a note may be moved to reach her own voice. Beyond this the shift
# starts to sound like a different person rather than a different note, and it
# is why the melody is transposed into her range instead of the voice being
# dragged across the whole keyboard to meet the melody.
MAX_SHIFT_SEMITONES = 7.0


@_needs_numpy
def estimate_f0(samples, sample_rate=DEFAULT_SAMPLE_RATE, low_hz=90.0, high_hz=400.0):
    """Strongest spectral peak inside a band.

    The band is deliberately narrow and caller supplied: an octave mistake is
    impossible when the search cannot reach the octave above or below, which is
    why the voice is measured in its natural range and a finished note is
    measured around the note it was asked for.
    """
    if samples.size < 1024:
        return 0.0
    span = min(samples.size, 4096)
    if samples.size > span:
        energy = np.convolve(
            np.abs(samples), np.ones(span // 4, dtype=np.float32), mode="valid"
        )
        start = int(np.argmax(energy))
    else:
        start = 0
    chunk = samples[start:start + span]
    if chunk.size < 512:
        return 0.0
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(chunk.size)))
    freqs = np.fft.rfftfreq(chunk.size, 1.0 / sample_rate)
    band = np.nonzero((freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0 - 1.0)))[0]
    if band.size < 3:
        return 0.0
    peak = band[int(np.argmax(spectrum[band]))]
    if float(spectrum[peak]) <= 0.0:
        return 0.0
    return float(freqs[peak])


def note_pitch(samples, semitone, sample_rate=DEFAULT_SAMPLE_RATE):
    """Measure a rendered note against the note it was supposed to be."""
    target = note_hz(semitone)
    return estimate_f0(samples, sample_rate, target / 1.7, target * 1.7)


_VOICE_BASE_HZ = None


@_needs_numpy
def voice_base_hz(synth, sample_rate=DEFAULT_SAMPLE_RATE):
    """Find Piper's natural pitch once, so a melody lands on real notes.

    Resampling only moves pitch relative to the voice she already has, so the
    target note has to be divided by that natural pitch rather than used alone.
    """
    global _VOICE_BASE_HZ
    if _VOICE_BASE_HZ:
        return _VOICE_BASE_HZ
    measured = 0.0
    try:
        raw = synth("la", 1.0)
    except Exception:
        raw = None
    if raw is not None and getattr(raw, "size", 0):
        measured = estimate_f0(trim_silence(raw, sample_rate), sample_rate)
    if not measured or not 80.0 <= measured <= 600.0:
        measured = FALLBACK_BASE_HZ
    _VOICE_BASE_HZ = measured
    return _VOICE_BASE_HZ


def _resample_to(samples, new_length):
    """Linear resample to an exact length."""
    new_length = max(1, int(new_length))
    if samples.size == new_length:
        return samples
    positions = np.linspace(0.0, samples.size - 1.0, new_length, dtype=np.float64)
    return np.interp(positions, np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)


def _pitch_marks(samples, period):
    """One mark per glottal cycle, each at the same phase of the cycle.

    A fixed grid of analysis positions is walked, and the strongest sample
    inside each window becomes the mark. Phase consistency is what lets the
    overlap-add below reuse a grain without the copy clicking.
    """
    step = max(2, int(round(period)))
    marks = []
    for start in range(0, max(1, samples.size - step), step):
        window = samples[start:start + step]
        if window.size < 2:
            break
        marks.append(start + int(np.argmax(np.abs(window))))
    if not marks:
        return np.zeros(0, dtype=np.int64)
    # Two marks can land in the same cycle when the pitch estimate is low, and
    # a grain repeated there is an echo rather than a note.
    kept = [marks[0]]
    for mark in marks[1:]:
        if mark - kept[-1] >= step * 0.6:
            kept.append(mark)
    return np.asarray(kept, dtype=np.int64)


def _psola_shift(samples, ratio, period):
    """Change pitch with time-domain PSOLA, which leaves the formants alone.

    Resampling a voice to move its pitch moves its formants by the same factor,
    so a note two semitones up is not the same singer on a higher note, it is a
    smaller singer. PSOLA instead keeps every grain at its original length and
    only re-spaces the glottal pulses, which is what a singer actually does, and
    it holds the length of the fragment so the note still fills its slot.

    Raising the pitch means the pulses come closer together, so the synthesis
    period is the analysis period divided by the ratio. Grains are two
    synthesis periods long and Hann-windowed, which is what keeps the sum from
    dipping between pulses.
    """
    ratio = float(ratio)
    if not math.isfinite(ratio):
        ratio = 1.0
    # A shift past these bounds is not a note any more, it is an artifact, so
    # the request is held inside them.
    ratio = max(MIN_PITCH_RATIO, min(MAX_PITCH_RATIO, ratio))
    period = float(period)
    if not math.isfinite(period) or period <= 0.0:
        return None
    synthesis_period = period / max(1e-3, ratio)
    half = max(2, int(round(synthesis_period / 2.0)))
    grain_length = 2 * half
    marks = _pitch_marks(samples, period)
    if marks.size < 2 or samples.size <= grain_length:
        return None

    output = np.zeros(samples.size + 2 * half, dtype=np.float32)
    weight = np.zeros(output.size, dtype=np.float32)
    window = np.hanning(grain_length).astype(np.float32)
    for position in np.arange(half, output.size - half, synthesis_period):
        centre = int(marks[int(np.argmin(np.abs(marks - position)))])
        start = centre - half
        end = start + grain_length
        if start < 0 or end > samples.size:
            continue
        target = int(round(position)) - half
        output[target:target + grain_length] += samples[start:end] * window
        weight[target:target + grain_length] += window

    voiced = weight > 1e-4
    if not voiced.any():
        return None
    shaped = np.zeros_like(output)
    shaped[voiced] = output[voiced] / weight[voiced]
    # The grains at the two ends are cut off by the edges of the buffer. Putting
    # the untouched original back there avoids a fade to nothing on every note.
    head = int(marks[0])
    shaped[:head] = samples[:head]
    tail = samples.size - int(marks[-1])
    if tail > 0:
        shaped[output.size - tail:] = samples[samples.size - tail:]
    return shaped[: samples.size]


def _measure_period(samples, sample_rate, prior_hz):
    """The fragment's own pulse period, measured rather than assumed.

    A pitch guess that is even slightly wrong moves every grain off the
    glottal pulses and detunes the whole note, so the fragment is measured
    first and the caller's estimate is only a fallback for a fragment too
    short to measure.
    """
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError):
        rate = DEFAULT_SAMPLE_RATE
    if not math.isfinite(rate) or rate <= 0.0:
        rate = DEFAULT_SAMPLE_RATE
    try:
        guess = float(prior_hz)
    except (TypeError, ValueError):
        guess = 0.0
    fallback = rate / max(60.0, min(500.0, guess or FALLBACK_BASE_HZ))

    count = int(getattr(samples, "size", 0))
    if count < 1024:
        return fallback
    window = samples[:8192] if count > 8192 else samples
    try:
        centred = np.asarray(window, dtype=np.float64) - float(np.mean(window))
    except (TypeError, ValueError):
        return fallback
    if centred.size < 1024:
        return fallback
    try:
        size = 1 << int(np.ceil(np.log2(2 * centred.size)))
        spectrum = np.fft.rfft(centred, size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[: centred.size]
    except (TypeError, ValueError, FloatingPointError, MemoryError):
        return fallback

    energy = float(correlation[0])
    if not math.isfinite(energy) or energy <= 0.0:
        return fallback
    correlation = correlation / energy
    low = max(2, int(rate / 400.0))
    high = min(correlation.size - 1, int(rate / 90.0))
    if high <= low or low >= correlation.size:
        return fallback
    lag = int(np.argmax(correlation[low:high])) + low
    # Only a clear peak is trusted: a weak one is the autocorrelation of an
    # unvoiced consonant, and trusting it detunes the note.
    if not math.isfinite(float(correlation[lag])) or correlation[lag] <= 0.3:
        return fallback
    return float(lag)


@_needs_numpy
def pitch_shift(samples, ratio, sample_rate=DEFAULT_SAMPLE_RATE, f0_hz=0.0):
    """Move a fragment onto its note without dragging its voice along with it."""
    try:
        shift = float(ratio)
    except (TypeError, ValueError):
        return samples
    if not math.isfinite(shift):
        return samples
    if samples.size == 0 or abs(shift - 1.0) < 1e-3:
        return samples
    # The same bound the PSOLA shift uses, so the resample fallback below
    # cannot stretch a fragment to an absurd length either.
    shift = max(MIN_PITCH_RATIO, min(MAX_PITCH_RATIO, shift))
    period = _measure_period(samples, sample_rate, f0_hz)
    shifted = None
    if samples.size > int(period * 3):
        try:
            shifted = _psola_shift(samples, shift, period)
        except (ValueError, IndexError, MemoryError):
            shifted = None
    if shifted is None or shifted.size == 0:
        # Too short to find a glottal cycle in, which is most one-syllable
        # fragments. Resampling keeps the note right and the voice wrong, which
        # still beats dropping the syllable.
        return _resample_to(samples, samples.size / shift)
    return shifted.astype(np.float32)


def _fit_length(text, slot_seconds, target_hz, synth, sample_rate, base_hz):
    """Synthesise a fragment as close to slot_seconds long as Piper can manage.

    Length is driven through Piper's own length_scale rather than a time
    stretch: overlapping-add detunes a voice, while Piper simply speaks slower,
    and the pitch that costs is measured and corrected straight afterwards.
    """
    wanted = max(0.09, estimate_seconds(text))
    guess = target_hz / max(40.0, float(base_hz or FALLBACK_BASE_HZ))
    length_scale = max(0.6, min(3.0, slot_seconds * guess / wanted))

    fragment = None
    measured_hz = 0.0
    for attempt in range(2):
        raw = synth(text, length_scale)
        if raw is None or raw.size == 0:
            return None, 0.0, length_scale
        fragment = trim_silence(raw, sample_rate)
        if fragment.size == 0:
            fragment = raw
        measured_hz = estimate_f0(fragment, sample_rate)
        current = fragment.size / sample_rate
        if current <= 0.01:
            return fragment, measured_hz, length_scale
        corrected = max(0.6, min(3.0, length_scale * slot_seconds / current))
        if abs(corrected - length_scale) <= 0.12:
            length_scale = corrected
            break
        length_scale = corrected
    return fragment, measured_hz, length_scale


@_needs_numpy
def add_vibrato(samples, sample_rate, rate_hz=5.0, depth_semitones=0.25):
    """Modulate pitch with a slow sine so held notes sound sung.

    The swing is centred on zero so the note stays on its pitch and only
    wobbles around it, and it only opens up over the second half of the note.
    A vibrato that is already at full depth on the attack is the wobble of a
    synthesiser; a singer holds the note straight first and lets it move later.
    """
    if samples.size == 0 or depth_semitones <= 0.0:
        return samples
    seconds = samples.size / sample_rate
    if seconds < 0.35:
        # Shorter than this is a syllable, not a note, and wobbling it just
        # makes the word harder to hear.
        return samples
    times = np.arange(samples.size, dtype=np.float64) / sample_rate
    onset = np.clip((times - seconds * 0.45) / max(1e-6, seconds * 0.35), 0.0, 1.0)
    swing = -0.5 * depth_semitones * onset * np.cos(2.0 * math.pi * rate_hz * times)
    read_positions = np.cumsum(np.power(2.0, swing))
    read_positions = np.clip(read_positions, 0.0, samples.size - 1.0)
    return np.interp(read_positions, np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)


@_needs_numpy
def add_breath(samples, sample_rate, seconds=0.09):
    """A short noisy onset before a phrase, so a line starts like a singer.

    Without one every line begins on a perfectly clean attack, which is the
    giveaway that no air went into the note.
    """
    if samples.size == 0 or seconds <= 0.0:
        return samples
    length = int(min(seconds * sample_rate, max(1, samples.size // 8)))
    if length < 8:
        return samples
    rng = np.random.default_rng(int(samples.size) % (2 ** 32))
    noise = rng.standard_normal(length).astype(np.float32)
    # Low-passed, so it sits under the vowel instead of hissing over it.
    kernel = np.hanning(length).astype(np.float32)
    kernel /= kernel.max() if kernel.max() > 0 else 1.0
    noise = np.convolve(noise, kernel, mode="same") * 0.06
    shaped = np.concatenate([noise, samples])
    return shaped.astype(np.float32)


@_needs_numpy
def fade_edges(samples, sample_rate, attack=0.02, release=0.05):
    """Short fades so note edges do not click."""
    if samples.size == 0:
        return samples
    shaped = samples.astype(np.float32).copy()
    attack_samples = min(shaped.size // 2, max(1, int(attack * sample_rate)))
    release_samples = min(shaped.size - attack_samples, max(1, int(release * sample_rate)))
    if attack_samples > 1:
        shaped[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
    if release_samples > 1:
        shaped[-release_samples:] *= np.linspace(1.0, 0.0, release_samples, dtype=np.float32)
    return shaped


@_needs_numpy
def trim_silence(samples, sample_rate, threshold=0.02):
    """Drop the quiet head and tail Piper pads onto every fragment."""
    if samples.size == 0:
        return samples
    window = max(1, int(0.005 * sample_rate))
    usable = samples.size - window
    if usable <= 0:
        return samples
    loudness = np.sqrt(
        np.convolve(samples.astype(np.float32) ** 2, np.ones(window, dtype=np.float32), mode="valid")
    )
    peak = float(loudness.max()) if loudness.size else 0.0
    if peak <= 0.0:
        return samples
    cut = threshold * peak
    loud_part = np.nonzero(loudness > cut)[0]
    if loud_part.size == 0:
        return samples
    start = int(loud_part[0])
    end = int(loud_part[-1]) + window
    return samples[start:min(end, samples.size)]


@_needs_numpy
def normalize(samples, peak=0.92):
    loudest = float(np.max(np.abs(samples))) if samples.size else 0.0
    if loudest <= 1e-6:
        return samples
    return (samples * (peak / loudest)).astype(np.float32)


@_needs_numpy
def join_notes(previous, following, sample_rate, seconds=0.018):
    """Overlap two notes so a line flows instead of stopping between words.

    A butt joint on two separately synthesised fragments is a step in the
    waveform, and a step is a click. A short crossfade welds them into the one
    breath a singer would have taken.
    """
    if previous.size == 0:
        return following
    if following.size == 0:
        return previous
    overlap = int(min(seconds * sample_rate, previous.size // 2, following.size // 2))
    if overlap < 4:
        return np.concatenate([previous, following])
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    welded = previous.copy()
    welded[-overlap:] = previous[-overlap:] * (1.0 - ramp) + following[:overlap] * ramp
    return np.concatenate([welded, following[overlap:]]).astype(np.float32)


@_needs_numpy
def render_note(text, semitone, slot_seconds, synth, sample_rate, tune, base_hz=None, legato=False):
    """Synthesise one fragment and bend it onto its note.

    Piper lowers its pitch when it is asked to speak slowly, so the note is
    measured after synthesis and shifted by whatever ratio that fragment
    actually needs instead of a ratio guessed up front.
    """
    target_hz = note_hz(semitone)
    fragment, measured_hz, length_scale = _fit_length(
        text, slot_seconds, target_hz, synth, sample_rate, base_hz
    )
    if fragment is None or fragment.size == 0:
        return None, sample_rate

    if not measured_hz or not 80.0 <= measured_hz <= 900.0:
        measured_hz = base_hz or voice_base_hz(synth, sample_rate)
    ratio = target_hz / max(40.0, measured_hz)

    shifted = pitch_shift(fragment, ratio, sample_rate, measured_hz)
    vibrato_rate, vibrato_depth = tune.get("vibrato", (5.0, 0.2))
    shifted = add_vibrato(shifted, sample_rate, vibrato_rate, vibrato_depth)
    if legato:
        # Inside a line the notes run into each other, so the only edges worth
        # shaping are the ones the crossfade in join_notes() cannot hide.
        shifted = fade_edges(shifted, sample_rate, attack=0.004, release=0.006)
    else:
        shifted = fade_edges(shifted, sample_rate)

    wanted_samples = int(min(slot_seconds * 1.3, max(0.08, slot_seconds)) * sample_rate)
    if shifted.size < wanted_samples:
        shifted = np.concatenate(
            [shifted, np.zeros(wanted_samples - shifted.size, dtype=np.float32)]
        )
    elif shifted.size > wanted_samples * 2:
        shifted = shifted[:wanted_samples * 2]

    return shifted, sample_rate


@_needs_numpy
def render_song(
    lines,
    synth,
    sample_rate=DEFAULT_SAMPLE_RATE,
    tune=None,
    tempo=None,
    base_semitone=BASE_SEMITONE,
    budget_seconds=DEFAULT_BUDGET_SECONDS,
    gap_seconds=0.12,
    report=None,
):
    """Build the whole song in memory and return float32 mono samples."""
    tune = tune or TUNES["sunny"]
    base_hz = voice_base_hz(synth, sample_rate)
    planned = fit_to_voice(
        plan_phrase(lines, tune, tempo=tempo, base_semitone=base_semitone), base_hz
    )
    if not planned:
        return np.zeros(0, dtype=np.float32), sample_rate

    gap = np.zeros(max(0, int(gap_seconds * sample_rate)), dtype=np.float32)
    pieces = []
    song_seconds = 0.0
    started = time.monotonic()
    for text, semitone, slot, starts_line in planned:
        if time.monotonic() - started > budget_seconds:
            break
        note, rate = render_note(
            text, semitone, slot, synth, sample_rate, tune, base_hz, legato=not starts_line
        )
        if rate and rate != sample_rate:
            sample_rate = rate
        if note is None or note.size == 0:
            continue
        if report is not None:
            report.append(
                {
                    "text": text,
                    "semitone": semitone,
                    "hz": round(note_hz(semitone), 1),
                    "shift": round(note_hz(semitone) / max(40.0, base_hz), 2),
                    "slot": round(slot, 3),
                    "seconds": round(note.size / sample_rate, 3),
                    "start": round(song_seconds, 3),
                }
            )
        song_seconds += note.size / sample_rate
        if starts_line:
            # A line is one breath: air, then the words, then a real pause.
            note = add_breath(note, sample_rate)
            if pieces:
                pieces.append(gap)
        if pieces:
            pieces[-1] = join_notes(pieces[-1], note, sample_rate)
        else:
            pieces.append(note)

    if not pieces:
        return np.zeros(0, dtype=np.float32), sample_rate

    song = np.concatenate(pieces).astype(np.float32)
    fade = min(int(0.05 * sample_rate), song.size // 4)
    if fade > 1:
        song[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        song[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return normalize(song), sample_rate


def describe(lyric_lines, emotion="neutral", intensity=0.6, song_name=""):
    """A short line about how a song will be sung, for her reply text."""
    name, tune = choose_tune(emotion, intensity, song_name)
    notes = plan_notes(lyric_lines, tune)
    seconds = sum(note[2] for note in notes) if notes else 0.0
    return {
        "tune": name,
        "tempo": tune.get("tempo"),
        "notes": len(notes),
        "seconds": round(seconds, 1),
    }