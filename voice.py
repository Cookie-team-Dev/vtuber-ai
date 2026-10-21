import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

log = logging.getLogger(__name__)

VOICE_MODEL = Path(__file__).resolve().with_name("en_US-amy-medium.onnx")
VOICE_ENABLED = True

try:
    from piper import PiperVoice, SynthesisConfig
    PIPER_AVAILABLE = True
except Exception:
    PiperVoice = None
    SynthesisConfig = None
    PIPER_AVAILABLE = False

try:
    import numpy as np
    NUMPY_OK = True
except Exception:
    np = None
    NUMPY_OK = False

if not VOICE_MODEL.exists():
    VOICE_ENABLED = False
    log.warning("Voice model %s not found; voice disabled.", VOICE_MODEL)

_STAGE_DIRECTION_PATTERN = re.compile(r"\*[^*]*\*")
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)
_DECORATION_PATTERN = re.compile(r"[~`^_=+<>|]+")
_CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_UNDERLINE_PATTERN = re.compile(r"__(.+?)__", re.DOTALL)
_STRIKETHROUGH_PATTERN = re.compile(r"~~(.+?)~~", re.DOTALL)
_BACKTICK_PATTERN = re.compile(r"`(.+?)`", re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_URL_PATTERN = re.compile(r"https?://\S+")
_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s*", flags=re.MULTILINE)
_BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s*", flags=re.MULTILINE)


def sanitize_for_speech(text):
    """Remove stage directions, markdown and emoji so Piper reads clean dialogue."""
    if not isinstance(text, str):
        return ""
    cleaned = _STAGE_DIRECTION_PATTERN.sub(" ", text)
    cleaned = _EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = _CODE_FENCE_PATTERN.sub(" ", cleaned)
    cleaned = _BOLD_PATTERN.sub(r"\1", cleaned)
    cleaned = _UNDERLINE_PATTERN.sub(r"\1", cleaned)
    cleaned = _STRIKETHROUGH_PATTERN.sub(r"\1", cleaned)
    cleaned = _BACKTICK_PATTERN.sub(r"\1", cleaned)
    cleaned = _HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _HEADING_PATTERN.sub("", cleaned)
    cleaned = _BLOCKQUOTE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\[([^\]]*)\]", r"\1", cleaned)
    cleaned = re.sub(r"\(([^)]*)\)", r"\1", cleaned)
    cleaned = _DECORATION_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


VOICE_LENGTH_SCALES = {
    "neutral": 1.0,
    "happy": 0.96,
    "sad": 1.08,
    "angry": 0.93,
    "surprised": 0.91,
    "shy": 1.03,
    "excited": 0.88,
    "curious": 0.96,
    "concerned": 1.05,
    "relieved": 0.98,
    "amused": 0.93,
    "grateful": 0.97,
    "affectionate": 1.0,
    "frustrated": 0.94,
    "hopeful": 0.95,
    "tired": 1.12,
    "calm": 1.04,
}


_AUDIO_PLAYER = None


def _find_audio_player():
    """Locate the playback command once; scanning PATH per line is wasteful."""
    global _AUDIO_PLAYER
    if _AUDIO_PLAYER is None:
        for name, args in (
            ("paplay", ()),
            ("aplay", ()),
            ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "error")),
        ):
            found = shutil.which(name)
            if found:
                _AUDIO_PLAYER = (found, args)
                break
        else:
            raise RuntimeError("Install paplay, aplay, or ffplay to play Piper audio.")
    return _AUDIO_PLAYER


def audio_player(audio_path):
    player, args = _find_audio_player()
    return [player, *args, audio_path]

_PIPER_VOICE = None
_PIPER_LOCK = threading.Lock()


def _piper_voice():
    """Load the Piper model once and keep it resident.

    Spawning `python -m piper` per line reloads the whole 63 MB ONNX model
    every single time; holding it in this process costs that once.
    """
    global _PIPER_VOICE
    if not PIPER_AVAILABLE:
        return None
    if _PIPER_VOICE is None:
        with _PIPER_LOCK:
            if _PIPER_VOICE is None:
                try:
                    _PIPER_VOICE = PiperVoice.load(str(VOICE_MODEL))
                except Exception:
                    return None
    return _PIPER_VOICE


def warm_up():
    """Load the voice model ahead of the first reply, off the critical path."""
    try:
        _piper_voice()
        return True
    except Exception:
        return False


def _synthesize_with_piper(text, wav_path, length_scale):
    """Synthesise in-process. Raises if the in-process API is unavailable."""
    if not PIPER_AVAILABLE:
        raise ModuleNotFoundError("piper module not installed")
    voice = _piper_voice()
    if voice is None:
        raise RuntimeError("piper voice unavailable")
    with wave.open(wav_path, "wb") as wav_file:
        voice.synthesize_wav(
            text, wav_file, syn_config=SynthesisConfig(length_scale=length_scale)
        )


def _synthesize_with_cli(text, wav_path, length_scale):
    if not PIPER_AVAILABLE:
        raise ModuleNotFoundError("piper module not installed")
    if not VOICE_MODEL.exists():
        raise FileNotFoundError(f"Voice model not found: {VOICE_MODEL}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper",
            "-m",
            str(VOICE_MODEL),
            "-f",
            wav_path,
            "--length-scale",
            f"{length_scale:.2f}",
            "-i",
            "/dev/stdin",
        ],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
        timeout=30,
        start_new_session=True,
    )


def speak(text, emotion="neutral", intensity=0.5):
    if not VOICE_ENABLED or not text or not text.strip():
        return

    spoken = sanitize_for_speech(text)
    if not spoken:
        return

    try:
        intensity = min(1.0, max(0.0, float(intensity)))
    except (TypeError, ValueError):
        intensity = 0.5
    target_scale = VOICE_LENGTH_SCALES.get(str(emotion).casefold(), 1.0)
    length_scale = 1.0 + (target_scale - 1.0) * intensity

    # Use a temp file that we control manually to avoid Windows file locking
    audio_path = None
    try:
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            _synthesize_with_piper(spoken, audio_path, length_scale)
        except Exception:
            # Any problem with the in-process API falls back to the CLI.
            try:
                _synthesize_with_cli(spoken, audio_path, length_scale)
            except subprocess.TimeoutExpired:
                log.warning("Piper CLI timed out while synthesizing speech")
                return
            except (subprocess.CalledProcessError, FileNotFoundError, ModuleNotFoundError):
                return
        try:
            # A player that never returns would stall the caller forever, and
            # the caller is the thread that owes the user a reply.
            subprocess.run(
                audio_player(audio_path),
                check=True,
                timeout=120,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            log.warning("Audio playback timed out")
        except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
            # RuntimeError covers "no audio player installed": never let a
            # silent machine break the reply itself.
            pass
    finally:
        if audio_path:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def _read_wav(path):
    """Return float32 mono samples and the sample rate of a wav file."""
    if not NUMPY_OK:
        return None, 0
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return np.zeros(0, dtype=np.float32), rate
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate


def _synthesize_pcm(text, length_scale=1.0, noise_scale=0.45, noise_w_scale=0.5):
    """Synthesise straight to samples, with settings that hold a note steady."""
    if not PIPER_AVAILABLE:
        return None, 0
    if not text or not text.strip():
        return None, 0
    voice = _piper_voice()
    buffer = io.BytesIO()
    try:
        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize_wav(
                text,
                wav_file,
                syn_config=SynthesisConfig(
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w_scale=noise_w_scale,
                ),
            )
    except Exception:
        return None, 0
    buffer.seek(0)
    try:
        with wave.open(buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            rate = wav_file.getframerate()
    except Exception:
        return None, 0
    if not NUMPY_OK or width != 2:
        return None, 0
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate


def _synthesize_fragment(text, length_scale=1.0):
    """Samples only, for the melody renderer."""
    samples, _rate = _synthesize_pcm(text, length_scale)
    if samples is not None and samples.size:
        return samples
    return _sing_fragment_via_cli(text, length_scale)


def _write_wav(path, samples, sample_rate):
    if not NUMPY_OK:
        return
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def sing_song(lyrics, emotion="happy", intensity=0.7, song_name=""):
    """Sing lyric lines on a real melody, one note per syllable."""
    if not VOICE_ENABLED or not lyrics:
        return
    lines = [line.strip() for line in lyrics if isinstance(line, str) and line.strip()]
    if not lines:
        return
    try:
        intensity = min(1.0, max(0.0, float(intensity)))
    except (TypeError, ValueError):
        intensity = 0.7

    try:
        import singing

        _, tune = singing.choose_tune(
            emotion, intensity, song_name or singing.guess_song_name(lines)
        )
    except Exception:
        _sing_lines_as_speech(lines, emotion, intensity)
        return

    audio_path = None
    try:
        samples, rate = singing.render_song(
            lines,
            _synthesize_fragment,
            tune=tune,
            base_semitone=singing.BASE_SEMITONE,
        )
        if samples.size == 0:
            _sing_lines_as_speech(lines, emotion, intensity)
            return
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        _write_wav(audio_path, samples, rate)
        try:
            subprocess.run(
                audio_player(audio_path),
                check=True,
                timeout=300,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            log.warning("Song playback timed out")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            pass
    except Exception:
        _sing_lines_as_speech(lines, emotion, intensity)
    finally:
        if audio_path:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def _sing_fragment_via_cli(text, length_scale):
    """Fall back to the Piper CLI when the in-process API is unavailable."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _synthesize_with_cli(text, path, length_scale)
        samples, _rate = _read_wav(path)
        return samples if samples.size else None
    except subprocess.TimeoutExpired:
        log.warning("Piper CLI timed out while singing fragment")
        return None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _sing_lines_as_speech(lines, emotion, intensity):
    """Last resort: read the lyrics out instead of dropping the song."""
    target_scale = VOICE_LENGTH_SCALES.get(str(emotion).casefold(), 1.0)
    sing_scale = max(1.2, 1.0 + (target_scale - 1.0) * intensity + 0.15)
    for line in lines:
        spoken = sanitize_for_speech(line)
        if not spoken:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            try:
                _synthesize_with_piper(spoken, audio_file.name, sing_scale)
            except Exception:
                try:
                    _synthesize_with_cli(spoken, audio_file.name, sing_scale)
                except subprocess.TimeoutExpired:
                    log.warning("Piper CLI timed out while singing lyrics")
                    continue
                except (subprocess.CalledProcessError, FileNotFoundError, ModuleNotFoundError):
                    continue
            try:
                subprocess.run(
                    audio_player(audio_file.name),
                    check=True,
                    timeout=120,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                log.warning("Audio playback timed out while singing lyrics")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
                pass
            time.sleep(0.15)
