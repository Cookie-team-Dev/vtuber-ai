import io
import importlib
import logging
import os
import queue as queue_module
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

from amika_tts_codes import find_sfx_file, parse_speech_codes, piper_emotion

log = logging.getLogger(__name__)

VOICE_MODEL = Path(__file__).resolve().with_name("en_US-amy-medium.onnx")
AMIKA_ENGINE = Path(__file__).resolve().parent / "openamika" / "vendor" / "tts-engine"
AMIKA_VOICE = AMIKA_ENGINE / "Speakers" / "MekaHime_Amika" / "character.json"
AMIKA_TTS_URL = os.environ.get("KOOKIE_AMIKA_TTS_URL", "http://127.0.0.1:8002").rstrip("/")
AMIKA_TTS_SPEED = float(os.environ.get("KOOKIE_AMIKA_SPEED", "1.2"))
if not 0.5 <= AMIKA_TTS_SPEED <= 2.0:
    raise ValueError("KOOKIE_AMIKA_SPEED must be between 0.5 and 2.0")

# One global speaking-rate multiplier on top of the per-emotion scales below.
# The emotion scales already push most moods under 1.0 (excited 0.88, angry 0.93)
# because a streamer's natural pace is quicker than neutral, and this used to
# be the only knob - so "speak faster" meant editing twelve numbers. It is now
# one env var, applied to the Piper length_scale and the OpenAmika speed in the
# same direction: lower is faster, for both. 0.85 is a 15% cut with the voices
# still intelligible; below about 0.75 the SoVITS decoder starts dropping
# syllables and it reads as rushed rather than quick.
TTS_SPEED = float(os.environ.get("KOOKIE_TTS_SPEED", "0.85"))
if not 0.5 <= TTS_SPEED <= 2.0:
    raise ValueError("KOOKIE_TTS_SPEED must be between 0.5 and 2.0")
VOICE_ENABLED = (
    VOICE_MODEL.exists()
    or AMIKA_VOICE.exists()
    or "KOOKIE_AMIKA_TTS_URL" in os.environ
)

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

if not VOICE_ENABLED:
    log.warning("Neither Piper weights nor OpenAmika voice assets are installed; voice disabled.")

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


_USE_PIPER_ONLY = os.environ.get("KOOKIE_TTS_BACKEND", "").lower() == "piper"
_AMIKA_WARNED = False
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


def _split_into_sentences(text):
    """Split text into speakable sentences without breaking numbers.

    A naive split on [.!?] mangles "22.5 degrees" into "22" and "5 degrees"
    and turns "Dr. Smith" into "Dr" and "Smith". The Laya voice assistant
    solved this by protecting the boundary first, so the pipeline below can
    start synthesizing the next sentence while the current one is still
    playing and never cut a number in half.
    """
    if not text:
        return []
    protected = text
    # A period between digits is a decimal, not a sentence end.
    protected = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", protected)
    # A period following a known abbreviation is not a sentence end either. The
    # boundary after the dot has to be a lookahead rather than \b: a dot next
    # to a space is not a word boundary, so \b silently failed and every
    # "Dr." was still treated as a sentence end.
    for abbrev in ("Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Lt",
                   "Capt", "Sgt", "Rev", "Hon", "e.g", "i.e", "vs", "etc",
                   "Inc", "Ltd", "Co", "Corp", "al", "a.m", "p.m"):
        protected = re.sub(
            r"\b" + abbrev + r"\.(?=\s|$)", abbrev + "<ABBREV>", protected
        )
    raw = re.split(r"(?<=[.!?])\s+", protected.strip())
    sentences = []
    for chunk in raw:
        chunk = chunk.replace("<DEC>", ".").replace("<ABBREV>", ".").strip()
        if chunk:
            sentences.append(chunk)
    return sentences


def _speak_pipelined(sentences, emotion, intensity, amika_emotion):
    """Play sentence N while synthesizing sentence N+1 in the background.

    The old path synthesized the whole reply, then played it, so time-to-first-
    audio was the full length of the TTS call. A person hears nothing until the
    entire line is ready. The pipeline keeps a single prefetch thread running
    ahead of the player: as soon as a sentence is synthesized it is queued,
    and the player starts the next WAV the instant the current one finishes.
    stop_speaking sets the stop flag, so a barge-in stops the player and the
    prefetch thread instead of waiting for the line to end.
    """
    if not sentences:
        return

    _PIPELINE_STOP.clear()
    queue = queue_module.Queue()

    def _prefetch():
        for sentence in sentences:
            if _PIPELINE_STOP.is_set():
                return
            audio_path = _synthesize_to_file(
                sentence, emotion, intensity, amika_emotion
            )
            if audio_path is None:
                continue
            queue.put(audio_path)
        queue.put(None)  # sentinel: synthesis is done

    prefetch_thread = threading.Thread(target=_prefetch, daemon=True)
    prefetch_thread.start()

    try:
        while True:
            item = queue.get()
            if item is None:
                break
            if _PIPELINE_STOP.is_set():
                try:
                    os.unlink(item)
                except OSError:
                    pass
                break
            try:
                _play_synthesized_audio(item)
            finally:
                try:
                    os.unlink(item)
                except OSError:
                    pass
    finally:
        _PIPELINE_STOP.set()
        prefetch_thread.join(timeout=2)


def speak(text, emotion="neutral", intensity=0.5):
    if not VOICE_ENABLED or not text or not text.strip():
        return

    try:
        intensity = min(1.0, max(0.0, float(intensity)))
    except (TypeError, ValueError):
        intensity = 0.5

    segments = parse_speech_codes(text, emotion)
    for kind, content, segment_emotion in segments:
        if kind == "sfx":
            sfx_path = find_sfx_file(content)
            if sfx_path is None:
                log.warning(
                    "Amika speech cue %s skipped; install a matching WAV in speech/static/sfx.",
                    content,
                )
                continue
            try:
                subprocess.run(
                    audio_player(str(sfx_path)),
                    check=True,
                    timeout=15,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                log.warning("Sound-effect playback timed out for %s", content)
            except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as error:
                log.warning("Could not play sound effect %s: %s", content, error)
            continue

        spoken = sanitize_for_speech(content)
        if not spoken:
            continue
        # A reply is usually several sentences. The pipeline starts the next
        # sentence's synthesis while the current one is still playing, so the
        # first sound arrives as soon as the first WAV is ready instead of at
        # the end of the whole line. A single sentence or a failure to split
        # falls back to the direct path, which is unchanged.
        sentences = _split_into_sentences(spoken)
        if len(sentences) <= 1:
            _speak_text(
                spoken,
                piper_emotion(segment_emotion),
                intensity,
                amika_emotion=segment_emotion,
            )
        else:
            _speak_pipelined(
                sentences,
                piper_emotion(segment_emotion),
                intensity,
                segment_emotion,
            )


def _synthesize_to_file(spoken, emotion, intensity, amika_emotion="neutral"):
    """Synthesize one chunk to a WAV path and return it, or None on failure.

    This is the shared body of _speak_text and the pipelined speak path: the
    pipeline needs to hand a finished WAV to a second thread while the first
    thread is still playing, so synthesis has to be a function that returns a
    path rather than one that plays inline. The caller owns the temp file and
    must unlink it.
    """
    target_scale = VOICE_LENGTH_SCALES.get(str(emotion).casefold(), 1.0)
    length_scale = (1.0 + (target_scale - 1.0) * intensity) * TTS_SPEED

    fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        try:
            _synthesize_with_amika(spoken, amika_emotion, audio_path)
        except Exception:
            global _AMIKA_WARNED
            if not _AMIKA_WARNED:
                _AMIKA_WARNED = True
                print("OpenAmika TTS unavailable; using Piper.", flush=True)
            try:
                _synthesize_with_piper(spoken, audio_path, length_scale)
            except Exception:
                try:
                    _synthesize_with_cli(spoken, audio_path, length_scale)
                except subprocess.TimeoutExpired:
                    log.warning("Piper CLI timed out while synthesizing speech")
                    return None
                except (subprocess.CalledProcessError, FileNotFoundError, ModuleNotFoundError):
                    return None
        return audio_path
    except Exception:
        log.warning("Synthesis failed for %r", spoken[:40], exc_info=True)
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        return None


def _speak_text(spoken, emotion, intensity, amika_emotion="neutral"):
    """Prefer OpenAmika GPT-SoVITS, falling back to Piper for this segment."""
    audio_path = _synthesize_to_file(spoken, emotion, intensity, amika_emotion)
    if audio_path is None:
        return
    try:
        _play_synthesized_audio(audio_path)
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def _synthesize_with_amika(text, emotion, wav_path):
    if _USE_PIPER_ONLY:
        raise RuntimeError("Piper-only mode")
    """Request one WAV segment from OpenAmika's local Socket.IO TTS gateway."""
    socketio = importlib.import_module("socketio")
    adapter = importlib.import_module("openamika.speech.voice_adapter")
    po_contract = importlib.import_module("openamika.speech.po_contract")
    enabled, character_id, emotion_map = adapter.live_state()
    request_text = f"【{emotion}】.{text}"
    if enabled:
        request_text = adapter.remap_leading_tag(request_text, emotion_map)
    language_code = po_contract.detect_lang(text)
    language = {"zh": "Chinese", "en": "English", "ja": "Japanese"}.get(
        language_code, "auto"
    )

    task_id = str(uuid.uuid4())
    completed = threading.Event()
    audio_events = []
    errors = []
    client = socketio.Client(reconnection=False, logger=False, engineio_logger=False)

    @client.on("audio_ready")
    def on_audio_ready(data):
        if not isinstance(data, dict) or data.get("task_id") != task_id:
            return
        audio_events.append(data)
        if data.get("is_final"):
            completed.set()

    @client.on("error")
    def on_error(data):
        if isinstance(data, dict) and data.get("task_id") == task_id:
            errors.append(str(data.get("message") or data.get("error") or data))
            completed.set()

    @client.on("disconnect")
    def on_disconnect(*_args):
        if not completed.is_set():
            errors.append("OpenAmika TTS connection closed before audio completed")
            completed.set()

    try:
        client.connect(AMIKA_TTS_URL, transports=["websocket"], wait_timeout=10)
        client.emit(
            "tts_request_gateway",
            {
                "task_id": task_id,
                "turn_id": task_id,
                "text": request_text,
                "character_id": character_id or "MekaHime_Amika",
                "language": language,
                "audio_format": "wav",
                # KOOKIE_AMIKA_SPEED is the raw engine factor; TTS_SPEED is the
                # global knob on top of it, so one env var moves both paths.
                "speed": AMIKA_TTS_SPEED * TTS_SPEED,
            },
        )
        if not completed.wait(timeout=300):
            raise TimeoutError("OpenAmika TTS request timed out")
        if errors:
            raise RuntimeError(f"OpenAmika TTS failed: {errors[0]}")
        if not any(isinstance(event.get("audio"), bytes) for event in audio_events):
            raise RuntimeError("OpenAmika TTS returned no audio")
        _write_amika_wav(audio_events, wav_path)
    finally:
        if client.connected:
            client.disconnect()


def _write_amika_wav(audio_events, wav_path):
    """Reassemble sentence/chunk events into one playable WAV file."""
    groups = []
    active_group = None
    for event in audio_events:
        audio = event.get("audio")
        if not isinstance(audio, bytes):
            continue
        chunk_index = event.get("chunk_index")
        chunk_count = event.get("chunk_count")
        if isinstance(chunk_index, int) and not isinstance(chunk_index, bool):
            if chunk_index == 0 or active_group is None:
                active_group = {"chunks": {}, "count": chunk_count or 1}
                groups.append(active_group)
            active_group["chunks"][chunk_index] = audio
            if len(active_group["chunks"]) >= active_group["count"]:
                active_group = None
        else:
            groups.append({"chunks": {0: audio}, "count": 1})
            active_group = None

    frames = []
    wav_params = None
    for group in groups:
        chunks = group["chunks"]
        expected_count = group["count"]
        if (
            not isinstance(expected_count, int)
            or expected_count < 1
            or len(chunks) != expected_count
            or set(chunks) != set(range(expected_count))
        ):
            raise ValueError("OpenAmika TTS returned incomplete audio chunks")
        segment = b"".join(chunks[index] for index in range(expected_count))
        with wave.open(io.BytesIO(segment), "rb") as wav_file:
            params = wav_file.getparams()
            segment_frames = wav_file.readframes(wav_file.getnframes())
        if wav_params is None:
            wav_params = params
        elif params[:3] != wav_params[:3]:
            raise ValueError("OpenAmika TTS returned incompatible WAV sentence formats")
        frames.append(segment_frames)

    if not frames or wav_params is None:
        raise RuntimeError("OpenAmika TTS returned no complete WAV audio")
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setparams(wav_params)
        wav_file.writeframes(b"".join(frames))


def synthesize_for_voice_chat(text, wav_path):
    """Create voice-chat audio with Amika first and Piper as a local fallback."""
    try:
        _synthesize_with_amika(text, "neutral", wav_path)
    except Exception as error:
        log.warning("OpenAmika voice-chat TTS unavailable; using Piper: %s", error)
        try:
            _synthesize_with_piper(text, wav_path, 1.0)
        except Exception as piper_error:
            log.warning("In-process Piper voice-chat TTS unavailable; using Piper CLI: %s", piper_error)
            _synthesize_with_cli(text, wav_path, 1.0)


def _play_synthesized_audio(audio_path):
    """Play one WAV, tracking the process so stop_speaking can kill it.

    The old version ran the player with check=True and swallowed every error,
    so a barge-in had nothing to kill and the line played out. The process is
    tracked here and released when it exits, so stop_speaking can cut it off
    mid-sentence instead of waiting for the sentence to finish.
    """
    try:
        proc = subprocess.Popen(
            audio_player(audio_path),
            start_new_session=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
        return
    _track_playback(proc)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        log.warning("Audio playback timed out")
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _clear_playback(proc)


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


_AMIKA_SERVER_PROC = None

# The currently-playing audio process, if any. stop_speaking kills it so a
# person can cut her off mid-sentence; without this she finishes the line.
_PLAYBACK_PROC = None
_PLAYBACK_LOCK = threading.Lock()

# The pipeline's stop flag. stop_speaking sets this so a barge-in halts both
# the player and the prefetch thread instead of waiting for the line to end.
_PIPELINE_STOP = threading.Event()


def _track_playback(proc):
    """Remember the active player so stop_speaking can kill just it."""
    global _PLAYBACK_PROC
    with _PLAYBACK_LOCK:
        _PLAYBACK_PROC = proc


def _clear_playback(proc):
    global _PLAYBACK_PROC
    with _PLAYBACK_LOCK:
        if _PLAYBACK_PROC is proc:
            _PLAYBACK_PROC = None


def stop_speaking():
    """Stop all speech playback immediately.

    This is the barge-in path: Mommy starts talking while Kookie is still
    speaking and the line has to die on the spot, not at the end of the
    sentence. The old body called interrupt(), which does not exist here, so
    every stop raised NameError and nothing was ever silenced; it also carried
    leftover fragments of sing_song, speak_cue and say_greeting that referenced
    parameters no caller passes, so even the first line was dead code.
    """
    global _PLAYBACK_PROC
    _PIPELINE_STOP.set()
    with _PLAYBACK_LOCK:
        proc = _PLAYBACK_PROC
        _PLAYBACK_PROC = None
    if proc is None:
        return
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
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
