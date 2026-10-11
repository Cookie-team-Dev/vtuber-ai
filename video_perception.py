"""Kookie's eyes and ears for video.

No hosted vision model and no API key, so watching a video means
measuring it with local models: faces (insightface, which reads
anime faces too, with YuNet as a backup), the colours that fill
the frame, how often the scene changes, how much moves, how
bright it is, what the frame is of (CLIP, asked about a few
hundred plain-word labels), who the people are (a gallery of
faces Mommy taught her), and the words on screen (easyocr).
Listening means real speech recognition: faster-whisper runs on
the CPU with the tiny model, which fits next to the 4B chat
model in the memory this box has left.

Every backend is optional. A missing model turns into an honest
sentence in the answer instead of an exception, because a tool
that crashes reads to the model like a dead tool. The heavy
models are loaded the first time she needs them and then kept:
Mommy said she does not care how much they eat.
"""

import gc
import os
import shutil
import subprocess
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "models")
_FACES_DIR = os.path.join(_MODELS_DIR, "faces")
_CHARACTERS_FILE = os.path.join(_MODELS_DIR, "characters.txt")
# tiny.en: English-only, ~75MB quantised. The big models need more RAM than
# this box has free next to the chat model, and Kookie's person speaks English.
_WHISPER_MODEL = os.environ.get("KOOKIE_WHISPER_MODEL", "tiny.en")
# Long videos are sampled, not sat through: the first two and a half minutes
# is enough to say what a video is about without spending five minutes of CPU.
_WHISPER_SECONDS = float(os.environ.get("KOOKIE_WHISPER_SECONDS", "150"))
# Whisper shares the box with Chrome and the chat model; two threads keeps
# it from starving them while still being several times realtime.
_WHISPER_THREADS = 2
# Below this much free RAM the transcription is skipped rather than
# risking the OOM killer taking the whole session down.
_MIN_WHISPER_RAM_MB = 350
# Gallery matches above this cosine are the same person. ArcFace is
# trained for roughly this point on real faces; lower and strangers
# start sharing names, higher and she never recognises anyone. A
# genuine match of the same photo measures about 0.99, so half way
# is a safe floor that still forgives a bad angle.
_FACE_MATCH = float(os.environ.get("KOOKIE_FACE_MATCH", "0.5"))
# ... and the winner has to beat the runner-up by this much,
# because two close scores mean she cannot tell the people
# apart, and mixing them up is worse than not naming either.
_FACE_AMBIGUITY = float(os.environ.get("KOOKIE_FACE_AMBIGUITY", "0.06"))
# A CLIP label only makes it into her words when it beats the
# runner-up by this much, so she does not narrate noise.
_LABEL_MARGIN = 0.02
# ... and a character name needs to win clearly, because guessing
# a name wrong is worse than not guessing.
_CHARACTER_MARGIN = 0.05

_INSIGHT = None
_YUNET = None
_WHISPER = None
_CLIP = None
_OCR = None
_GALLERY = None


def memory_available_mb():
    """Free RAM the kernel still has to give, in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 9999.0


# ----------------------------- eyes: faces -----------------------------


def _insight():
    """The insightface pack (detection, landmarks, pose, age, gender,
    recognition), loaded once. It reads anime faces, which is why it
    is the primary detector rather than a backup."""
    global _INSIGHT
    if _INSIGHT is not None:
        return _INSIGHT
    _INSIGHT = False
    try:
        import numpy as np  # noqa: F401 - insightface needs it loaded first

        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _INSIGHT = app
    except Exception:
        _INSIGHT = False
    return _INSIGHT


def _yunet():
    """The YuNet face detector, loaded once; None when it cannot run."""
    global _YUNET
    if _YUNET is not None:
        return _YUNET
    _YUNET = None
    try:
        import cv2

        path = os.path.join(_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
        if not os.path.exists(path):
            return _YUNET
        _YUNET = cv2.FaceDetectorYN.create(
            path, "", (320, 320), score_threshold=0.6
        )
    except Exception:
        _YUNET = None
    return _YUNET


def _merge_overlaps(boxes, iou_limit=0.3):
    """Drop duplicate boxes: the same face found twice would
    read as two people."""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        duplicate = False
        for other in kept:
            ix = max(0.0, min(box[0] + box[2], other[0] + other[2]) - max(box[0], other[0]))
            iy = max(0.0, min(box[1] + box[3], other[1] + other[3]) - max(box[1], other[1]))
            overlap = ix * iy
            if overlap > iou_limit * min(box[2] * box[3], other[2] * other[3]):
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def _record_from_insight(face, width, height):
    """One insightface hit as a plain record, or None when it is
    too small to be worth a sentence. Small faces are mostly
    background people; counting every one of them drowns the
    answer in names that are not there."""
    x1, y1, x2, y2 = face["bbox"]
    w, h = x2 - x1, y2 - y1
    if w < width * 0.02 or h < height * 0.02:
        return None
    record = {
        "box": [x1 / width, y1 / height, w / width, h / height],
        "score": float(face.get("det_score") or 0.0),
        "age": None,
        "gender": None,
        "pose": None,
        "smile": None,
        "embedding": face.get("normed_embedding") or face.get("embedding"),
    }
    age = face.get("age")
    if age:
        record["age"] = int(age)
    gender = face.get("gender")
    if gender is not None:
        record["gender"] = "female" if int(gender) == 1 else "male"
    pose = getattr(face, "pose", None)
    if pose is not None:
        # insightface gives [pitch, yaw, roll] in degrees: yaw is
        # the turn to the side, pitch is up and down.
        record["pose"] = {"pitch": float(pose[0]), "yaw": float(pose[1]), "roll": float(pose[2])}
    record["smile"] = _smile_score(face)
    return record


def _smile_score(face):
    """How much the mouth corners are pulled apart, compared with
    how far apart the eyes are. A real smile widens the mouth
    past the eyes; a closed mouth never does."""
    try:
        lm = face.get("landmark_2d_106")
        if lm is None:
            return None
        import numpy as np

        # The 106 points put the mouth around 76-95 and the eyes
        # around 33-50; the exact indices are stable across the
        # model's releases, and a wrong guess only costs a smile.
        mouth = lm[76:96]
        eyes = lm[33:51]
        mouth_spread = float(np.linalg.norm(mouth[0] - mouth[-1]))
        eye_spread = float(np.linalg.norm(eyes[0] - eyes[-1]))
        if eye_spread <= 0:
            return None
        return mouth_spread / eye_spread
    except Exception:
        return None


def _record_from_yunet(face, width, height):
    x, y, w, h = face[0], face[1], face[2], face[3]
    if w < width * 0.02 or h < height * 0.02:
        return None
    return {
        "box": [x / width, y / height, w / width, h / height],
        "score": float(face[-1]),
        "age": None,
        "gender": None,
        "pose": None,
        "smile": None,
        "embedding": None,
    }


def find_faces(image, max_width=1280):
    """Face records for everyone in the frame: box, pose, smile,
    age and gender when the model is confident enough to guess,
    and the embedding used to tell one person from another."""
    app = _insight()
    try:
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)
        img = image.convert("RGB")
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)))
        import numpy as np

        arr = np.asarray(img)
        records = []
        if app:
            for face in app.get(arr):
                record = _record_from_insight(face, img.width, img.height)
                if record:
                    records.append(record)
        if not records:
            detector = _yunet()
            if detector is not None:
                detector.setInputSize((img.width, img.height))
                _found, faces = detector.detect(arr)
                if faces is not None:
                    for face in faces:
                        record = _record_from_yunet(face, img.width, img.height)
                        if record:
                            records.append(record)
        return _merge_overlaps_records(records)
    except Exception:
        return []


def _same_person(one, other, iou_limit=0.3):
    """True when two sightings are the same person.

    Same gallery name settles it outright; so does a strong
    embedding match, which survives a change of angle; and
    as a last resort, the same place in the frame - which
    is why the embedding rules come first: two different
    people can stand in the same spot at different times.
    """
    if one.get("name") and one.get("name") == other.get("name"):
        return True
    if one.get("embedding") is not None and other.get("embedding") is not None:
        try:
            import numpy as np

            a = np.asarray(one["embedding"], dtype=np.float32)
            b = np.asarray(other["embedding"], dtype=np.float32)
            if float(a @ b) > 0.65:
                return True
        except Exception:
            pass
    x1, y1, w1, h1 = one["box"]
    x2, y2, w2, h2 = other["box"]
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap = ix * iy
    return overlap > iou_limit * min(w1 * h1, w2 * h2)


def _dedupe_people(records):
    """One entry per person: she stands in several of the
    sampled moments, and her name once per moment would
    read like a crowd."""
    deduped = []
    for record in records:
        for index, kept in enumerate(deduped):
            if _same_person(record, kept):
                # The clearer sighting wins, so a blurred
                # early frame does not hide a sharp later one.
                if record["score"] > kept["score"]:
                    deduped[index] = record
                break
        else:
            deduped.append(record)
    return deduped


def _merge_overlaps_records(records, iou_limit=0.3):
    """Same idea as _merge_overlaps, for rich records."""
    records = sorted(records, key=lambda r: r["box"][2] * r["box"][3], reverse=True)
    kept = []
    for record in records:
        box = record["box"]
        duplicate = False
        for other in kept:
            o = other["box"]
            ix = max(0.0, min(box[0] + box[2], o[0] + o[2]) - max(box[0], o[0]))
            iy = max(0.0, min(box[1] + box[3], o[1] + o[3]) - max(box[1], o[1]))
            overlap = ix * iy
            if overlap > iou_limit * min(box[2] * box[3], o[2] * o[3]):
                duplicate = True
                break
        if not duplicate:
            kept.append(record)
    return kept


# ----------------------------- eyes: who -----------------------------


def _slug(name):
    keep = [c if c.isalnum() or c in " -_" else "_" for c in str(name)]
    return "".join(keep).strip() or "person"


def _face_gallery():
    """The people Mommy taught her: (name, average embedding) pairs.

    One folder per person under models/faces, any number of photos
    in it. Averaging several photos of the same person is what
    makes the match survive a bad haircut or a different camera.
    """
    global _GALLERY
    if _GALLERY is not None:
        return _GALLERY
    _GALLERY = []
    try:
        if not os.path.isdir(_FACES_DIR):
            return _GALLERY
        import numpy as np

        app = _insight()
        if not app:
            return _GALLERY
        for person in sorted(os.listdir(_FACES_DIR)):
            folder = os.path.join(_FACES_DIR, person)
            photos = []
            if os.path.isdir(folder):
                for root, _dirs, files in os.walk(folder):
                    for name in sorted(files):
                        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            photos.append(os.path.join(root, name))
            elif person.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                photos = [folder]
                person = os.path.splitext(person)[0]
            embeddings = []
            for photo in photos:
                try:
                    from PIL import Image

                    img = Image.open(photo).convert("RGB")
                    faces = find_faces(img)
                    # The largest face in a photo of a person is
                    # that person; a group photo teaches her the
                    # wrong name otherwise.
                    if faces:
                        best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                        if best["embedding"] is not None:
                            embeddings.append(np.asarray(best["embedding"], dtype=np.float32))
                except Exception:
                    continue
            if embeddings:
                average = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(average)
                if norm > 0:
                    _GALLERY.append((person, average / norm))
    except Exception:
        _GALLERY = []
    return _GALLERY


def _match_face(embedding):
    """The gallery name for a face, when she is sure enough."""
    if embedding is None:
        return None
    try:
        import numpy as np

        gallery = _face_gallery()
        if not gallery:
            return None
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        vector = vector / norm
        best_name = None
        best_score = 0.0
        runner_score = 0.0
        for name, known in gallery:
            score = float(vector @ known)
            if score > best_score:
                best_name, best_score, runner_score = (
                    name,
                    score,
                    best_score,
                )
            elif score > runner_score:
                runner_score = score
        if (
            best_score >= _FACE_MATCH
            and best_score - runner_score >= _FACE_AMBIGUITY
        ):
            return best_name
        return None
    except Exception:
        return None


def remember_face(target):
    """Teach her a face: remember_face(name|photo path).

    The photo is kept under models/faces/ so she still knows the
    person after a restart, and the gallery is rebuilt right away
    so she can use it in the same conversation.
    """
    target = str(target or "")
    name, sep, path = target.partition("|")
    name = name.strip()
    path = path.strip()
    if not name or not sep or not path:
        return "Error: remember_face needs 'name|photo path'"
    if not os.path.isfile(path):
        return f"She could not find a photo at {path}."
    try:
        from PIL import Image

        faces = find_faces(Image.open(path).convert("RGB"))
    except Exception:
        faces = []
    if not faces:
        return "She could not find a face in that photo, so she will not remember it."
    folder = os.path.join(_FACES_DIR, _slug(name))
    os.makedirs(folder, exist_ok=True)
    base = os.path.basename(path) or "photo.jpg"
    if not base.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        base += ".jpg"
    target_path = os.path.join(folder, base)
    # Every photo she is given is kept: several
    # pictures of one person make a steadier
    # average than a single one, and the same
    # file name twice is a different photo, not
    # a replacement.
    if os.path.exists(target_path):
        stem = os.path.splitext(base)[0]
        ext = os.path.splitext(base)[1]
        counter = 1
        while os.path.exists(target_path):
            target_path = os.path.join(
                folder, f"{stem}-{counter}{ext}"
            )
            counter += 1
    shutil.copy(path, target_path)
    # A dozen pictures is plenty; past that the
    # oldest are dropped so the gallery rebuild
    # stays quick.
    keep = 12
    photos = sorted(
        (
            os.path.join(folder, name_)
            for name_ in os.listdir(folder)
            if name_.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        ),
        key=os.path.getmtime,
    )
    for old in photos[:-keep]:
        try:
            os.remove(old)
        except Exception:
            pass
    global _GALLERY
    _GALLERY = None
    return f"She will remember {name}'s face now."


def forget_face_gallery():
    """Drop the in-memory gallery; it is rebuilt from disk on use."""
    global _GALLERY
    _GALLERY = None


# ----------------------------- eyes: what -----------------------------


# Plain words CLIP is asked about. She does not see "a sunset", she
# sees whichever of these the picture is closest to, so the list is
# her whole vocabulary: anything not in it she cannot say. Style gets
# its own pass because a style word and a content word in one pool
# compete, and 'a fox' should not beat 'an anime screenshot' just
# because both are in the frame's neighbourhood.
STYLE_LABELS = (
    "an anime screenshot", "a cartoon", "a drawing", "a painting",
    "a sketch", "a comic book panel", "a 3d render", "a video game scene",
    "a 3d animated film", "a 2d animated film", "a stop motion film",
    "a photograph", "a live action scene", "black and white",
)
CONTENT_LABELS = (
    "a man", "a woman", "a girl", "a boy", "a child", "an old person",
    "a baby", "a crowd of people", "a person from behind",
    "a cat", "a dog", "a bird", "a horse", "a rabbit", "a fish",
    "a bear", "a fox", "a mouse", "a snake", "a dragon", "a monster",
    "a bedroom", "a kitchen", "a classroom", "a city street", "a forest",
    "a beach", "mountains", "the sky", "outer space", "underwater",
    "a restaurant", "an office", "a park", "inside a car", "a train",
    "a hospital", "a shop", "a castle", "a rooftop", "a battlefield",
    "someone running", "someone jumping", "someone singing", "a fight",
    "someone eating", "someone sleeping", "people talking",
    "someone dancing", "someone crying", "someone laughing",
    "a kiss", "a hug", "someone reading", "someone driving",
    "a concert", "a sports game", "someone cooking",
    "a car", "a tree", "food", "a book", "a phone", "a computer",
    "a sword", "a gun", "a guitar", "a piano", "a bicycle", "a boat",
    "a house", "a clock", "a television", "flowers", "clothes",
    "a bag", "a bottle", "a cup", "a mirror", "a robot",
    "a spaceship", "magic", "a superhero", "a mask",
    "a dark scene", "a bright scene", "a sunset", "night time",
    "a colourful scene", "snow", "rain", "fire", "smoke", "clouds",
    "water", "a storm", "a sunny day",
    "someone waving their hands", "someone pointing", "someone clapping",
    "someone gesturing", "a person's hands", "someone typing",
    "someone presenting", "a person talking to a camera",
    "a close-up of a face", "a magical girl", "a mecha robot",
    "a samurai", "a ninja", "an anime character",
)


def _clip():
    """CLIP, loaded once: the model that turns a picture and a
    sentence into numbers that can be compared.

    The openai weights, not the laion ones: on a long label
    list the laion ViT-B-32 ranks 'smoke' above 'a woman' on
    a portrait, because its similarities stay too close
    together to tell neighbours apart.
    """
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    _CLIP = False
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        _CLIP = (model, preprocess)
    except Exception:
        _CLIP = False
    return _CLIP


def clip_labels(images, labels=CONTENT_LABELS, top=3, margin=_LABEL_MARGIN):
    """What the frames are of, as (label, score) pairs.

    Every frame is asked about every label and the answers are
    averaged, because a label that only one frame agrees with is
    usually a coincidence.
    """
    clip = _clip()
    if not clip or not images:
        return []
    try:
        import torch

        model, preprocess = clip
        batch = torch.stack([preprocess(img.convert("RGB")) for img in images])
        with torch.no_grad():
            image_features = model.encode_image(batch)
            text_features = model.encode_text(
                __import__("open_clip").tokenize(list(labels))
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Mean over frames first: one clear frame should not
            # outvote several quiet ones.
            similarity = (image_features.mean(dim=0, keepdim=True)
                          @ text_features.t())[0]
        pairs = sorted(zip(similarity.tolist(), labels), reverse=True)
        result = []
        for score, label in pairs:
            if result and score < result[0][1] - margin:
                break
            result.append((label, score))
            if len(result) >= top:
                break
        return result
    except Exception:
        return []


def known_characters():
    """The characters Mommy listed, one name per line in
    models/characters.txt. CLIP knows a handful of famous ones
    already, but the list is how she learns the rest."""
    try:
        if not os.path.isfile(_CHARACTERS_FILE):
            return []
        with open(_CHARACTERS_FILE) as f:
            return [line.strip() for line in f if line.strip()][:80]
    except Exception:
        return []


_CHARACTER_TEXT_CACHE = {}

# The captioner, loaded on demand: BLIP says what a
# picture shows as a sentence, which is what she
# actually says out loud. A label list is the
# fallback for when the box cannot spare the RAM.
_CAPTION = None


def _captioner():
    """BLIP, loaded once and only when it fits.

    The caption model is the biggest thing her eyes
    load, about a gigabyte, so it stays out entirely
    on a starved box rather than making everything
    else swap."""
    global _CAPTION
    if _CAPTION is not None:
        return _CAPTION
    _CAPTION = False
    if memory_available_mb() < 900:
        return _CAPTION
    try:
        import torch

        from transformers import BlipForConditionalGeneration
        from transformers import BlipProcessor

        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model.eval()
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _CAPTION = (model, processor, torch)
    except Exception:
        _CAPTION = False
    return _CAPTION


def release_captioner():
    """Let the captioner go: a gigabyte is a lot here."""
    global _CAPTION
    if _CAPTION:
        _CAPTION = None
        gc.collect()


def release_vision_models():
    """Let the heavy eye models go, for a second.

    CLIP, the face pack, the small detector and the
    text reader each hold hundreds of megabytes. The
    captioner needs that room to load, and nothing
    after a caption needs them: they load again on
    the next picture, which is cheaper than holding
    everything at once on a box this size."""
    global _INSIGHT, _CLIP, _OCR, _YUNET
    _INSIGHT = None
    _CLIP = None
    _OCR = None
    _YUNET = None
    gc.collect()


def caption_image(image):
    """One sentence about what a single picture shows."""
    captioner = _captioner()
    if not captioner:
        return ""
    model, processor, torch = captioner
    try:
        inputs = processor(image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=3,
                no_repeat_ngram_size=2,
                repetition_penalty=1.15,
            )
        text = processor.decode(out[0], skip_special_tokens=True).strip()
        words = text.split()
        if len(words) < 4:
            return ""
        # A caption that keeps repeating itself has
        # degenerated and says nothing at all.
        if len(set(words)) < len(words) * 0.6:
            return ""
        return text[:140]
    except Exception:
        return ""


def _character_text_features(names):
    """The prompt word features for the character list.

    Encoding the prompts is the expensive half of a
    character guess, and the list does not change between
    frames, so it is built once per list of names and
    kept for the next video she watches.
    """
    import open_clip

    import torch

    key = tuple(names)
    cached = _CHARACTER_TEXT_CACHE.get(key)
    if cached is not None:
        return cached
    # Three ways of asking, because the one CLIP heard
    # most in training is not the one a cartoon frame
    # answers best.
    labels = (
        [f"an anime character named {n}" for n in names]
        + [f"a character named {n}" for n in names]
        + [f"a photo of {n}" for n in names]
    )
    model, _ = _clip()
    with torch.no_grad():
        features = model.encode_text(open_clip.tokenize(labels))
        features = features / features.norm(dim=-1, keepdim=True)
    _CHARACTER_TEXT_CACHE[key] = (labels, features)
    return labels, features


def clip_characters(images, names):
    """Which of the listed characters the frames might show.

    Deliberately hedged in the caller: CLIP has met these
    characters in the wild, not in a lineup, so a match is
    a guess she is allowed to be wrong about. A name only
    counts when it beats the runner-up by a wide margin,
    because naming the wrong character is worse than none.
    """
    if not names or not images:
        return []
    clip = _clip()
    if not clip:
        return []
    try:
        import torch

        model, preprocess = clip
        batch = torch.stack([preprocess(img.convert("RGB")) for img in images])
        labels, text_features = _character_text_features(names)
        with torch.no_grad():
            image_features = model.encode_image(batch)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            similarity = (
                image_features.mean(dim=0, keepdim=True) @ text_features.t()
            )[0]
        # The best of the prompt shapes per name.
        best = {}
        for index, label in enumerate(labels):
            for name in names:
                if label.endswith(name):
                    score = float(similarity[index])
                    if name not in best or score > best[name]:
                        best[name] = score
                    break
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        if not ranked:
            return []
        top_name, top_score = ranked[0]
        if len(ranked) > 1 and top_score - ranked[1][1] < _CHARACTER_MARGIN:
            return []
        return [top_name]
    except Exception:
        return []


# ----------------------------- eyes: words -----------------------------


def _ocr():
    """The on-screen text reader, loaded once."""
    global _OCR
    if _OCR is not None:
        return _OCR
    _OCR = False
    try:
        import easyocr

        _OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        _OCR = False
    return _OCR


def ocr_text(image, max_lines=3):
    """Words written on the frame, or None when there are none."""
    reader = _ocr()
    if not reader:
        return None
    try:
        import numpy as np
        from PIL import Image as PILImage

        img = image.convert("RGB")
        if img.width > 1280:
            img = img.resize((1280, int(img.height * 1280 / img.width)))
        results = reader.readtext(np.asarray(img), detail=1, paragraph=False)
        lines = []
        for box, text, confidence in results:
            text = str(text).strip()
            if text and float(confidence) > 0.4:
                lines.append(text)
        seen = []
        for line in lines:
            if line not in seen:
                seen.append(line)
            if len(seen) >= max_lines:
                break
        return "; ".join(seen) if seen else None
    except Exception:
        return None


# ----------------------------- eyes: measuring -----------------------------


# Nearest-name colour matching. Neutral greys are left out on purpose:
# brightness already says dark or bright, and "grey" next to a real
# colour only waters the sentence down.
_COLOR_NAMES = (
    ((255, 0, 0), "red"),
    ((255, 140, 0), "orange"),
    ((255, 255, 0), "yellow"),
    ((0, 128, 0), "green"),
    ((0, 200, 200), "cyan"),
    ((0, 0, 255), "blue"),
    ((148, 0, 211), "purple"),
    ((255, 105, 180), "pink"),
    ((139, 69, 19), "brown"),
)


def _name_color(rgb):
    best = None
    best_dist = None
    for center, name in _COLOR_NAMES:
        dist = sum((int(rgb[i]) - center[i]) ** 2 for i in range(3))
        if best_dist is None or dist < best_dist:
            best, best_dist = name, dist
    return best


def dominant_colors(image, top=3):
    """The colours that fill the frame, by name, most used first."""
    try:
        import numpy as np

        small = image.convert("RGB").resize((32, 32))
        arr = np.asarray(small, dtype=np.uint8).reshape(-1, 3)
        # 8 bins per channel: fine enough to tell blue from green, coarse
        # enough that one shade of skin does not steal every bin.
        quantized = arr // 32
        keys, counts = np.unique(quantized, axis=0, return_counts=True)
        order = counts.argsort()[::-1]
        names = []
        for idx in order:
            r, g, b = keys[idx] * 32 + 16
            if max(r, g, b) - min(r, g, b) < 24:
                continue
            name = _name_color((r, g, b))
            if name not in names:
                names.append(name)
            if len(names) >= top:
                break
        return names
    except Exception:
        return []


def sample_video_frames(path, max_frames=6):
    """Evenly spaced frames from a local video as (timestamp, PIL image)."""
    try:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        duration = total / fps if fps > 0 else 0
        if duration <= 0:
            # No reliable duration: fall back to walking the whole video.
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
        if duration <= 0:
            duration = 1.0
        count = min(max_frames, max(3, int(duration / 10) + 1))
        frames = []
        if total > 0:
            # Frame seeking is exact, and the last readable frame sits
            # one or two frames before the reported end - seeking to the
            # very end reads nothing back.
            last = max(0, total - 2)
            for i in range(count):
                frame_id = min(int(total * i / (count - 1)), last)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if ok:
                    frames.append(
                        (frame_id / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    )
        else:
            edge = max(0.0, duration - 0.2)
            for i in range(count):
                ts = min(duration * i / (count - 1), edge)
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ok, frame = cap.read()
                if ok:
                    frames.append((ts, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        cap.release()
        return frames
    except Exception:
        return []


def analyze_frames(images, read_text=True):
    """Measure a list of PIL images: who is in them, what they are
    of, and everything the cheap measurements can say."""
    import numpy as np

    face_records = []
    face_counts = []
    colors = []
    smalls = []
    for image in images:
        records = find_faces(image)
        for record in records:
            record["name"] = _match_face(record["embedding"])
        face_records.extend(records)
        face_counts.append(len(records))
        colors.extend(dominant_colors(image, top=2))
        smalls.append(np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32))
    people = _dedupe_people(face_records)

    motion = 0.0
    scene_changes = 0
    if len(smalls) > 1:
        diffs = []
        for a, b in zip(smalls, smalls[1:]):
            diffs.append(float(np.abs(a - b).mean()))
            ha = np.histogram(a, bins=16, range=(0, 255))[0].astype(np.float32)
            hb = np.histogram(b, bins=16, range=(0, 255))[0].astype(np.float32)
            norm = np.linalg.norm(ha) * np.linalg.norm(hb)
            corr = float((ha * hb).sum() / norm) if norm > 0 else 1.0
            if corr < 0.72:
                scene_changes += 1
        motion = float(np.mean(diffs))

    brightness = float(np.mean([s.mean() for s in smalls])) if smalls else 0.0
    color_counts = {}
    for name in colors:
        color_counts[name] = color_counts.get(name, 0) + 1
    top_colors = [name for name, _ in sorted(color_counts.items(), key=lambda kv: -kv[1])][:3]

    style = clip_labels(images, STYLE_LABELS, top=1)
    content = clip_labels(images, CONTENT_LABELS, top=2)
    style_anime = bool(style) and any(
        word in style[0][0]
        for word in ("anime", "cartoon", "drawing", "paint", "sketch", "comic", "animated", "render", "stop motion")
    )
    characters = []
    if style_anime:
        # A character name is only asked of a drawing or an
        # animation: on real flesh the same prompt would
        # name a stranger, and the encoding of the whole
        # list is saved on top of it.
        characters = clip_characters(images, known_characters())
    # What she actually says starts here: a sentence
    # about the first and the middle moment. The
    # models that already did their work are let go
    # first so the captioner fits alongside them.
    captions = []
    try:
        if images:
            release_vision_models()
            if len(images) > 1:
                picks = (
                    (0, "the start"),
                    (len(images) // 2, "the middle"),
                )
            else:
                picks = ((0, ""),)
            for index, place in picks:
                text = caption_image(images[index])
                if text:
                    captions.append((place, text))
    finally:
        release_captioner()
    # Age and gender are guesses about real flesh; on a drawing they
    # are noise, so a cartoon frame keeps its mouth shut about both.
    if style_anime:
        for record in people:
            record["age"] = None
            record["gender"] = None
    on_screen_text = None
    if read_text:
        on_screen_text = ocr_text(images[0]) if images else None

    return {
        "moments": len(images),
        "faces": people,
        "face_counts": face_counts,
        "colors": top_colors,
        "brightness": brightness,
        "motion": motion,
        "scene_changes": scene_changes,
        "labels": content,
        "style": style,
        "characters": characters,
        "captions": captions,
        "style_anime": style_anime,
        "ocr": on_screen_text,
    }


def _plural(count, one, many):
    return one if count == 1 else many


def _describe_face(record):
    """One person, as much as she can tell about them."""
    parts = []
    name = record.get("name")
    if name:
        parts.append(name)
    else:
        gender = record.get("gender")
        age = record.get("age")
        if gender and age:
            parts.append(
                f"a {gender} who looks to be in her {age // 10 * 10}s"
                if gender == "female"
                else f"a {gender} who looks to be in his {age // 10 * 10}s"
            )
        elif gender:
            parts.append(f"a {gender}")
        else:
            parts.append("a person")
    pose = record.get("pose")
    if pose:
        if pose["yaw"] > 20:
            parts.append("looking to the side")
        elif pose["yaw"] < -20:
            parts.append("looking the other way")
        if pose["pitch"] > 15:
            parts.append("looking down")
        elif pose["pitch"] < -15:
            parts.append("looking up")
        if abs(pose["roll"]) > 12:
            parts.append("head tilted")
    smile = record.get("smile")
    if smile is not None and smile > 0.95:
        parts.append("smiling")
    box = record["box"]
    if box[2] > 0.35:
        parts.append("close up")
    return " ".join(parts)


def describe_visuals(report):
    """Turn measurements into what she would say out loud."""
    faces = report.get("faces") or []
    face_counts = report.get("face_counts") or []
    parts = []

    # The sentence about what is shown comes first,
    # because that is the answer she was asked for;
    # everything after it is the detail around it.
    for place, text in report.get("captions") or []:
        if place:
            parts.append(f"at {place} it shows {text}")
        else:
            parts.append(f"I saw {text}")

    if face_counts:
        at_once = max(face_counts)
        with_faces = sum(1 for c in face_counts if c > 0)
        if at_once == 0:
            parts.append("I do not see any people's faces in it")
        else:
            who = _plural(at_once, "one person's face", f"{at_once} people's faces")
            where = ""
            boxes = [f["box"] for f in faces]
            if boxes:
                middle = sum(b[0] + b[2] / 2 for b in boxes) / len(boxes)
                if middle < 0.38:
                    where = ", mostly on the left side of the frame"
                elif middle > 0.62:
                    where = ", mostly on the right side of the frame"
                else:
                    where = ", mostly in the middle of the frame"
            parts.append(
                f"I can see {who}{where}"
                + (f", in {with_faces} of the {len(face_counts)} moments I watched"
                   if with_faces < len(face_counts) else "")
            )
            if faces:
                described = [_describe_face(record) for record in faces[:4]]
                parts.append("there is " + "; ".join(described))
                unknown = [r for r in faces if not r.get("name")]
                if unknown and len(unknown) < len(faces):
                    parts.append(
                        f"and {len(unknown)} more I do not know"
                    )

    labels = [label for label, _score in report.get("labels") or []]
    if labels:
        listed = ", ".join(labels[:2]) if len(labels) > 1 else labels[0]
        # On a drawing her content words are guesses, and a
        # guess said flatly is a lie; on a photograph they
        # are close to certain.
        if report.get("style_anime"):
            parts.append(f"it might show {listed}")
        else:
            parts.append(f"it looks like {listed}")

    style = [label for label, _score in report.get("style") or []]
    if style:
        parts.append(f"the style is {style[0]}")

    characters = report.get("characters") or []
    if characters:
        parts.append(f"it might be {characters[0]}")

    colors = report.get("colors") or []
    if colors:
        listed = ", ".join(colors[:2]) if len(colors) > 1 else colors[0]
        parts.append(f"the colours are mostly {listed}")

    brightness = report.get("brightness", 0)
    if brightness < 40:
        parts.append("it is dark")
    elif brightness < 100:
        parts.append("the lighting is dim")
    elif brightness > 190:
        parts.append("it is brightly lit")
    else:
        parts.append("the lighting is normal")

    scene_changes = report.get("scene_changes", 0)
    if scene_changes >= 3:
        parts.append(f"the scene changes about {scene_changes} times")
    elif scene_changes > 0:
        parts.append("the scene changes a little")

    motion = report.get("motion", 0)
    if motion > 14:
        parts.append("there is a lot of movement")
    elif motion > 5:
        parts.append("there is steady movement")
    elif motion > 1.5:
        parts.append("it is fairly still")
    else:
        parts.append("the picture barely moves")

    text = report.get("ocr")
    if text:
        parts.append(f'the words on screen say: "{text[:200]}"')

    return "; ".join(parts) + "."


def describe_local_video(path):
    """Watch a file on disk: who and what is in it."""
    if not os.path.isfile(str(path)):
        return f"She could not find a video file at {path}."
    frames = sample_video_frames(path)
    if not frames:
        return "She could not open that video file."
    report = analyze_frames([img for _, img in frames])
    count = report["moments"]
    return (
        f"I watched {count} "
        f"{_plural(count, 'moment', 'moments')} of it. "
        + describe_visuals(report)
    )


def describe_images(images):
    """Describe frames captured from a browser page."""
    if not images:
        return "She watched the page but caught no picture of the video."
    # A page shot carries the site's own words all over it, which
    # would drown out the video, so the reader stays off here and
    # the page's own description text says what the site claims.
    report = analyze_frames(images, read_text=False)
    count = report["moments"]
    return (
        f"I watched {count} "
        f"{_plural(count, 'moment', 'moments')} of the picture. "
        + describe_visuals(report)
    )


# ----------------------------- ears -----------------------------


def _run(args, timeout=180):
    return subprocess.run(
        args, capture_output=True, timeout=timeout
    )


def extract_audio(source, out_wav, seconds=None):
    """Turn any local media into a 16kHz mono wav for the ears."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(source)]
    if seconds:
        cmd += ["-t", str(int(seconds))]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", out_wav]
    result = _run(cmd)
    if result.returncode != 0 or not os.path.exists(out_wav):
        return None
    return out_wav


def download_audio(url, dest_dir, seconds=None):
    """Pull the audio track out of a web video (YouTube and friends)."""
    try:
        import yt_dlp

        raw = os.path.join(dest_dir, "audio_src")
        opts = {
            "format": "bestaudio/best",
            "outtmpl": raw,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if not info:
            return None, ""
        title = str(info.get("title") or "")
        if not os.path.exists(raw):
            raw = raw + "." + str(info.get("ext", "webm"))
        wav = os.path.join(dest_dir, "audio.wav")
        if extract_audio(raw, wav, seconds):
            return wav, title
        return None, title
    except Exception:
        return None, ""


def _whisper_model():
    """The speech recogniser, loaded once; False when it cannot run here."""
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    _WHISPER = False
    if memory_available_mb() < _MIN_WHISPER_RAM_MB:
        return _WHISPER
    try:
        from faster_whisper import WhisperModel

        _WHISPER = WhisperModel(
            _WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=_WHISPER_THREADS,
            download_root=_MODELS_DIR,
        )
    except Exception:
        _WHISPER = False
    return _WHISPER


def _load_audio_f32(wav_path):
    """Read a 16kHz mono wav as float32 samples in [-1, 1].

    faster-whisper's own file decoder needs a PyAV version whose API
    moves between releases, so the wav - which ffmpeg already produced
    at exactly the format whisper wants - is read here and handed over
    as samples instead.
    """
    import wave

    import numpy as np

    with wave.open(str(wav_path)) as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None, rate
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples), rate


def transcribe(wav_path, seconds=_WHISPER_SECONDS, deadline=240.0):
    """What is said out loud, or None when speech recognition is unavailable."""
    model = _whisper_model()
    if not model:
        return None
    samples, rate = _load_audio_f32(wav_path)
    if samples is None:
        return None
    if rate != 16000:
        # whisper only accepts 16kHz; ffmpeg already made it 16kHz, so
        # this only fires on a hand-made file. A straight line between
        # samples is enough for a format that should not happen.
        wanted = int(len(samples) * 16000 / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, wanted),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
    audio = samples[: int(seconds * 16000)]
    try:
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        pieces = []
        started = time.time()
        for segment in segments:
            pieces.append(segment.text.strip())
            if segment.end >= seconds or time.time() - started > deadline:
                break
        text = " ".join(pieces).strip()
        return text or None
    except Exception:
        # Some builds need the VAD model downloaded separately; a failure
        # there should not cost the whole transcription.
        try:
            segments, _info = model.transcribe(audio, language="en", beam_size=1)
            pieces = []
            started = time.time()
            for segment in segments:
                pieces.append(segment.text.strip())
                if segment.end >= seconds or time.time() - started > deadline:
                    break
            text = " ".join(pieces).strip()
            return text or None
        except Exception:
            return None


def describe_sound(wav_path):
    """The shape of the sound: loudness, quiet parts, tones versus talking."""
    try:
        import wave

        import numpy as np

        with wave.open(str(wav_path)) as w:
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) < rate // 10:
            return None
        win = max(1, rate // 20)  # 50ms windows
        windows = samples[: len(samples) // win * win].reshape(-1, win)
        rms = np.sqrt((windows ** 2).mean(axis=1))
        dbfs = 20.0 * np.log10(rms + 1e-9)
        mag = np.abs(np.fft.rfft(windows, axis=1)) + 1e-9
        flatness = np.exp(np.log(mag).mean(axis=1)) / mag.mean(axis=1)

        silent = dbfs < -50
        loud = dbfs >= -50
        # Low flatness means a steady tone (music, a held note); high
        # flatness means changing sound (talking, rustling).
        tonal = loud & (flatness < 0.08)
        varying = loud & (flatness >= 0.08)
        total = len(windows)
        loudest = int(np.argmax(dbfs)) * win / rate
        return {
            "duration": len(samples) / rate,
            "loudness": float(dbfs[loud].mean()) if loud.any() else -99.0,
            "silence_frac": float(silent.mean()),
            "tonal_frac": float(tonal.sum() / total),
            "varying_frac": float(varying.sum() / total),
            "loudest_at": loudest,
        }
    except Exception:
        return None


def _describe_sound_words(sound):
    parts = []
    if sound is None:
        return "she could not make sense of the sound"
    silence = sound["silence_frac"]
    if silence > 0.9:
        parts.append("it is almost silent")
    else:
        if silence > 0.4:
            parts.append("a lot of it is quiet")
        loud = sound["loudness"]
        if loud > -12:
            parts.append("it is loud")
        elif loud > -30:
            parts.append("the volume is normal")
        else:
            parts.append("it is quiet")
        tonal = sound["tonal_frac"]
        varying = sound["varying_frac"]
        if tonal > varying * 1.5 and tonal > 0.25:
            parts.append("the sound is mostly steady tones, like music")
        elif varying > tonal * 1.5 and varying > 0.25:
            parts.append("the sound keeps changing, like talking")
        elif tonal > 0.2 and varying > 0.2:
            parts.append("there is both music and changing sound, like talking over music")
    return ", ".join(parts)


def listen_to_wav(wav_path, seconds=_WHISPER_SECONDS):
    """The full hearing report for an audio file: words plus the sound."""
    text = transcribe(wav_path, seconds=seconds)
    sound = describe_sound(wav_path)
    parts = []
    if text:
        shown = text if len(text) <= 400 else text[:397] + "..."
        parts.append(f'what is said: "{shown}"')
    else:
        parts.append("she could not hear any words in it")
    words = _describe_sound_words(sound)
    if words:
        parts.append(words)
    return ". ".join(parts) + "."


def listen_to_file(path):
    """Listen to a video file on disk."""
    if not os.path.isfile(str(path)):
        return f"She could not find a video file at {path}."
    tmp = tempfile.mkdtemp(prefix="kookie_hears_")
    try:
        wav = extract_audio(path, os.path.join(tmp, "audio.wav"), seconds=_WHISPER_SECONDS)
        if not wav:
            return "There is no sound track in this video, so she cannot listen to it."
        return listen_to_wav(wav)
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def listen_to_url(url):
    """Listen to a web video by pulling its audio track down first."""
    tmp = tempfile.mkdtemp(prefix="kookie_hears_")
    try:
        wav, title = download_audio(url, tmp, seconds=_WHISPER_SECONDS)
        if not wav:
            return "She could not get the sound out of that link."
        report = listen_to_wav(wav)
        if title:
            return f"'{title}' - {report}"
        return report
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def looks_like_url(target):
    return "://" in str(target or "")
