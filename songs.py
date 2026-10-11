"""The songs Kookie knows by heart, and how she finds one from what Mommy said.

Everything in here is her own memory of the words: traditional songs, nursery
rhymes and old tunes that are out of copyright and free for anyone to sing. None
of it is downloaded, so a song she can sing is a song she knows without asking
anybody, which is what makes her useful when a link arrives with no lyrics.

Tunes are named, not defined: the melody itself belongs to singing.py, because
that is where notes are turned into sound.
"""

import re

# Only the words and the tune name per song. Keeping the library flat means a new
# song is one entry, not a new class.
SONGS = {
    "happy birthday": {
        "lyrics": [
            "Happy birthday to you",
            "Happy birthday to you",
            "Happy birthday dear Mommy",
            "Happy birthday to you",
        ],
        "tune": "sunny",
        "emotion": "happy",
        "duration": 3.0,
        "aliases": ["happy birthday to you", "birthday song"],
    },
    "twinkle twinkle little star": {
        "lyrics": [
            "Twinkle twinkle little star",
            "How I wonder what you are",
            "Up above the world so high",
            "Like a diamond in the sky",
            "Twinkle twinkle little star",
            "How I wonder what you are",
        ],
        "tune": "gentle",
        "emotion": "curious",
        "duration": 4.0,
        "aliases": ["twinkle twinkle", "twinkle", "ah vous dirai je", "the alphabet song"],
    },
    "rock a bye baby": {
        "lyrics": [
            "Rock a bye baby on the treetop",
            "When the wind blows the cradle will rock",
            "When the bough breaks the cradle will fall",
            "And Mommy will catch you, that's all",
        ],
        "tune": "gentle",
        "emotion": "calm",
        "duration": 4.0,
        "aliases": ["lullaby", "rock a bye", "rockabye baby", "hush little baby"],
    },
    "you are my sunshine": {
        "lyrics": [
            "You are my sunshine, my only sunshine",
            "You make me happy when skies are grey",
            "You'll never know dear, how much I love you",
            "Please don't take my sunshine away",
        ],
        "tune": "sunny",
        "emotion": "affectionate",
        "duration": 4.0,
        "aliases": ["my sunshine", "youre my sunshine"],
    },
    "mary had a little lamb": {
        "lyrics": [
            "Mary had a little lamb",
            "Little lamb, little lamb",
            "Mary had a little lamb",
            "Its fleece was white as snow",
            "And everywhere that Mary went",
            "The lamb was sure to go",
        ],
        "tune": "silly",
        "emotion": "happy",
        "duration": 3.0,
        "aliases": ["mary had a lamb"],
    },
    "row row row your boat": {
        "lyrics": [
            "Row row row your boat",
            "Gently down the stream",
            "Merrily merrily merrily",
            "Life is but a dream",
        ],
        "tune": "sunny",
        "emotion": "happy",
        "duration": 3.0,
        "aliases": ["row row row", "row your boat", "gently down the stream"],
    },
    "this old man": {
        "lyrics": [
            "This old man, he played one",
            "He played knick knack on my thumb",
            "With a knick knack paddy whack",
            "Give the dog a bone",
        ],
        "tune": "silly",
        "emotion": "amused",
        "duration": 3.0,
        "aliases": ["the wise old man", "this old man he played one"],
    },
    "itsy bitsy spider": {
        "lyrics": [
            "The itsy bitsy spider climbed up the water spout",
            "Down came the rain and washed the spider out",
        ],
        "tune": "silly",
        "emotion": "amused",
        "duration": 3.0,
        "aliases": ["itsy bitsy", "eensy weensy spider", "the itsy bitsy spider"],
    },
    "head shoulders knees and toes": {
        "lyrics": [
            "Head shoulders knees and toes",
            "Knees and toes",
            "Head shoulders knees and toes",
            "Knees and toes",
            "Eyes and ears and mouth and nose",
        ],
        "tune": "silly",
        "emotion": "happy",
        "duration": 3.0,
        "aliases": ["head shoulders", "head and shoulders"],
    },
    "old macdonald had a farm": {
        "lyrics": [
            "Old MacDonald had a farm",
            "E I E I O",
            "And on his farm he had a cow",
            "With a moo moo here and a moo moo there",
        ],
        "tune": "silly",
        "emotion": "amused",
        "duration": 3.0,
        "aliases": ["old macdonald", "macdonald had a farm"],
    },
    "baa baa black sheep": {
        "lyrics": [
            "Baa baa black sheep have you any wool",
            "Yes sir yes sir three bags full",
        ],
        "tune": "gentle",
        "emotion": "calm",
        "duration": 3.0,
        "aliases": ["black sheep", "baa baa"],
    },
    "hickory dickory dock": {
        "lyrics": [
            "Hickory dickory dock",
            "The mouse ran up the clock",
            "The clock struck one",
            "The mouse ran down",
        ],
        "tune": "silly",
        "emotion": "amused",
        "duration": 3.0,
        "aliases": ["hickory dickory", "hickory dickory dock"],
    },
    "pat a cake": {
        "lyrics": [
            "Pat a cake pat a cake baker's man",
            "Bake me a cake as fast as you can",
            "Pat it and roll it and mark it with B",
            "Put it in the oven for baby and me",
        ],
        "tune": "sunny",
        "emotion": "happy",
        "duration": 3.0,
        "aliases": ["patty cake", "pat-a-cake", "bakers man"],
    },
    "ring o roses": {
        "lyrics": [
            "Ring o' roses, ring o' roses",
            "A pocket full of posies",
        ],
        "tune": "gentle",
        "emotion": "calm",
        "duration": 3.0,
        "aliases": ["ring of roses", "ring o roses", "pocket full of posies"],
    },
    "london bridge": {
        "lyrics": [
            "London bridge is falling down",
            "Falling down, falling down",
            "London bridge is falling down",
            "My fair lady",
        ],
        "tune": "gentle",
        "emotion": "curious",
        "duration": 3.0,
        "aliases": ["london bridge is falling down", "londonbridge"],
    },
    "three blind mice": {
        "lyrics": [
            "Three blind mice, see how they run",
            "They all ran after the farmer's gun",
            "Who chopped off their tails",
        ],
        "tune": "silly",
        "emotion": "amused",
        "duration": 3.0,
        "aliases": ["three blind mice", "blind mice"],
    },
    "auld lang syne": {
        "lyrics": [
            "Should auld acquaintance be forgot",
            "And never brought to mind",
            "Should auld acquaintance be forgot",
            "And days of auld lang syne",
        ],
        "tune": "gentle",
        "emotion": "affectionate",
        "duration": 4.0,
        "aliases": ["auld lang syne", "auld acquaintances", "should auld acquaintance"],
    },
    "amazing grace": {
        "lyrics": [
            "Amazing grace how sweet the sound",
            "That saved a wretch like me",
            "I once was lost but now am found",
            "Was blind but now I see",
        ],
        "tune": "gentle",
        "emotion": "grateful",
        "duration": 4.0,
        "aliases": ["amazing grace", "how sweet the sound"],
    },
    "silent night": {
        "lyrics": [
            "Silent night holy night",
            "All is calm all is bright",
            "Round yon virgin mother and child",
            "Holy infant so tender and mild",
        ],
        "tune": "gentle",
        "emotion": "calm",
        "duration": 4.0,
        "aliases": ["silent night", "holy night", "stille nacht"],
    },
    "we wish you a merry christmas": {
        "lyrics": [
            "We wish you a merry christmas",
            "We wish you a merry christmas",
            "We wish you a merry christmas",
            "And a happy new year",
        ],
        "tune": "sunny",
        "emotion": "happy",
        "duration": 3.5,
        "aliases": ["merry christmas", "we wish you a merry christmas"],
    },
    "jingle bells": {
        "lyrics": [
            "Jingle bells jingle bells",
            "Jingle all the way",
            "Oh what fun it is to ride in a one horse open sleigh",
        ],
        "tune": "bright",
        "emotion": "excited",
        "duration": 3.5,
        "aliases": ["jingle bells", "one horse open sleigh"],
    },
    "scarborough fair": {
        "lyrics": [
            "Are you going to Scarborough fair",
            "Parsley sage rosemary and thyme",
            "Remember me to one who lives there",
            "She once was a true love of mine",
        ],
        "tune": "gentle",
        "emotion": "affectionate",
        "duration": 4.0,
        "aliases": ["scarborough fair", "parsley sage rosemary and thyme"],
    },
    "greensleeves": {
        "lyrics": [
            "Alas my love you do me wrong",
            "To sing and sing and never be in song",
        ],
        "tune": "gentle",
        "emotion": "sad",
        "duration": 3.5,
        "aliases": ["greensleeves", "green sleeves"],
    },
    "danny boy": {
        "lyrics": [
            "Oh Danny boy the pipes the pipes are calling",
            "From glen to glen and down the mountain sides",
        ],
        "tune": "dramatic",
        "emotion": "sad",
        "duration": 4.0,
        "aliases": ["danny boy", "londonderry air", "the pipes are calling"],
    },
    "the drunken sailor": {
        "lyrics": [
            "What shall we do with the drunken sailor",
            "What shall we do with the drunken sailor",
            "Early in the morning",
        ],
        "tune": "dramatic",
        "emotion": "amused",
        "duration": 4.0,
        "aliases": ["drunken sailor", "what shall we do with a drunken sailor"],
    },
    "home sweet home": {
        "lyrics": [
            "Home sweet home",
            "There's no place like home sweet home",
        ],
        "tune": "gentle",
        "emotion": "affectionate",
        "duration": 3.0,
        "aliases": ["home sweet home", "theres no place like home"],
    },
    "she'll be coming round the mountain": {
        "lyrics": [
            "She'll be coming round the mountain when the mountain turns red",
            "And the sky is blue in the morning",
        ],
        "tune": "sunny",
        "emotion": "happy",
        "duration": 3.5,
        "aliases": ["coming round the mountain", "round the mountain"],
    },
    "for shes a jolly good fellow": {
        "lyrics": [
            "For she's a jolly good fellow",
            "For she's a jolly good fellow",
            "And so I am too",
        ],
        "tune": "sunny",
        "emotion": "happy",
        "duration": 3.0,
        "aliases": ["jolly good fellow", "for shes a jolly good fellow"],
    },
    "yankee doodle": {
        "lyrics": [
            "Yankee doodle went to town",
            "Riding on a pound of pumpkins",
        ],
        "tune": "silly",
        "emotion": "amused",
        "duration": 3.0,
        "aliases": ["yankee doodle", "pound of pumpkins"],
    },
    "god rest ye merry gentlemen": {
        "lyrics": [
            "God rest ye merry gentlemen",
            "Let nothing you disturb",
        ],
        "tune": "gentle",
        "emotion": "calm",
        "duration": 3.5,
        "aliases": ["merry gentlemen", "god rest ye"],
    },
    "beautiful dreamer": {
        "lyrics": [
            "Beautiful dreamer wake up",
            "Dreaming of love that may never be",
        ],
        "tune": "gentle",
        "emotion": "affectionate",
        "duration": 3.5,
        "aliases": ["beautiful dreamer"],
    },
}

# A lyric that gives a song away even when the title does not.
_FINGERPRINTS = {
    "twinkle": "twinkle twinkle little star",
    "how i wonder": "twinkle twinkle little star",
    "rock a bye": "rock a bye baby",
    "rockabye": "rock a bye baby",
    "cradle will rock": "rock a bye baby",
    "you are my sunshine": "you are my sunshine",
    "mary had a little lamb": "mary had a little lamb",
    "its fleece was white": "mary had a little lamb",
    "row row row": "row row row your boat",
    "merrily merrily": "row row row your boat",
    "life is but a dream": "row row row your boat",
    "should auld acquaintance": "auld lang syne",
    "days of auld lang syne": "auld lang syne",
    "amazing grace": "amazing grace",
    "saved a wretch": "amazing grace",
    "all is calm": "silent night",
    "holy night": "silent night",
    "we wish you a merry": "we wish you a merry christmas",
    "jingle all the way": "jingle bells",
    "one horse open sleigh": "jingle bells",
    "scarborough fair": "scarborough fair",
    "parsley sage rosemary": "scarborough fair",
    "alas my love you do me wrong": "greensleeves",
    "danny boy": "danny boy",
    "pipes the pipes are calling": "danny boy",
    "drunken sailor": "the drunken sailor",
    "no place like home": "home sweet home",
    "round the mountain": "she'll be coming round the mountain",
    "jolly good fellow": "for shes a jolly good fellow",
    "pound of pumpkins": "yankee doodle",
    "merry gentlemen": "god rest ye merry gentlemen",
    "beautiful dreamer": "beautiful dreamer",
    "baa baa black sheep": "baa baa black sheep",
    "hickory dickory": "hickory dickory dock",
    "pat a cake": "pat a cake",
    "ring o roses": "ring o roses",
    "london bridge": "london bridge",
    "three blind mice": "three blind mice",
    "itsy bitsy spider": "itsy bitsy spider",
    "head shoulders knees": "head shoulders knees and toes",
    "old macdonald": "old macdonald had a farm",
    "this old man": "this old man",
    "happy birthday to you": "happy birthday",
}

# Streaming titles carry everything but the name of the song.
_NOISE_PATTERN = re.compile(
    r"\b("
    r"remaster(?:ed)?|remix|live|version|edit|mix|mono|stereo|deluxe|bonus|"
    r"explicit|clean|radio|single|album|remastered\s*\d{4}|"
    r"\d{4}\s*remaster|feat\.?[^,]*|ft\.?[^,]*|with\s+[^,]*|"
    r"official\s+(music\s+)?video|official\s+audio|lyrics?|hd|hq|4k"
    r")\b",
    re.IGNORECASE,
)
_BRACKET_PATTERN = re.compile(r"[\(\[\-][^\)\]]*[\)\]]")


def normalise_title(text):
    """A title with the packaging taken off, so two spellings meet."""
    text = str(text or "").strip().casefold()
    text = _BRACKET_PATTERN.sub(" ", text)
    text = _NOISE_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def titles_for(name):
    """Every spelling that should find a song: its name plus its aliases."""
    entry = SONGS.get(str(name or "").strip().casefold())
    if not entry:
        return []
    return [name, *entry.get("aliases", [])]


def find_song(query):
    """The song she knows that best matches what was said, or None."""
    wanted = normalise_title(query)
    if not wanted:
        return None

    for name in SONGS:
        if normalise_title(name) == wanted:
            return name
    for name, entry in SONGS.items():
        if any(normalise_title(alias) == wanted for alias in entry.get("aliases", [])):
            return name

    # A title can carry more than she asked for ("row row row your boat lullaby"),
    # so an alias inside the request counts too - the other way round would let
    # one short alias swallow every song.
    best = None
    for name, entry in SONGS.items():
        for candidate in (name, *entry.get("aliases", [])):
            needle = normalise_title(candidate)
            if len(needle) >= 6 and needle in wanted:
                if best is None or len(needle) > len(normalise_title(best[0])):
                    best = (name, entry)
    return best[0] if best else None


def guess_from_lyrics(lines):
    """Name the song from its words, for when Mommy sends the words themselves."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    joined = " ".join(str(line) for line in lines).casefold()
    joined = re.sub(r"[^a-z ]+", " ", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    for fingerprint, name in _FINGERPRINTS.items():
        if fingerprint in joined:
            return name
    return ""


def lyrics_for(name):
    return list(SONGS[name]["lyrics"]) if name in SONGS else []


def song_entry(name):
    return dict(SONGS[name]) if name in SONGS else None


def list_songs():
    return sorted(SONGS)


def summary_line(name):
    return f"{name} ({SONGS[name]['emotion']})" if name in SONGS else ""


def random_song(rng=None):
    import random as _random

    picker = rng or _random
    return picker.choice(list(SONGS))


# One syllable per note is what makes a hum follow the tune: the singing pipeline
# spends exactly one note on every piece it is given.
_HUM_WORDS = ("ooh", "aah", "oh", "ah")


def hum_lines(name, notes=16, rng=None):
    """Wordless lines that follow a song's tune, for singing with no lyrics."""
    entry = SONGS.get(str(name or "").strip().casefold())
    if not entry:
        return []
    picker = rng or __import__("random")
    word = picker.choice(_HUM_WORDS)
    lines = []
    for start in range(0, max(4, int(notes)), 4):
        lines.append(" ".join([word] * min(4, max(4, int(notes)) - start)))
    return lines


def hum_name_for(query):
    """The tune to hum for a request, matched just like a song to sing."""
    return find_song(query)


# The real tune for the songs whose melody is the whole point of singing them.
# One note name per syllable, C4 in the middle of her range, which is also where
# singing.py counts from. A song that is not listed keeps the generic pentatonic
# tune: a made-up "traditional" melody would be worse than an honest generic one.
#
# singing.py checks each list against the syllables of the lyrics above and only
# uses it when they line up, so a wording change can never leave notes hanging on
# the wrong syllable.
MELODIES = {
    "twinkle twinkle little star": {
        "tempo": 96,
        "vibrato": (5.0, 0.24),
        "lines": [
            "C4 C4 G4 G4 A4 A4 G4",
            "F4 F4 E4 E4 D4 D4 C4",
            "G4 G4 F4 F4 E4 E4 D4",
            "G4 G4 F4 F4 E4 E4 D4",
            "C4 C4 G4 G4 A4 A4 G4",
            "F4 F4 E4 E4 D4 D4 C4",
        ],
    },
    "happy birthday": {
        "tempo": 100,
        "vibrato": (5.2, 0.22),
        "lines": [
            "G4 G4 A4 G4 C5 B4",
            "G4 G4 A4 G4 D5 C5",
            "G4 G4 G5 E5 C5 B4 A4",
            "F4 F4 E4 C5 D5 C5",
        ],
    },
    "mary had a little lamb": {
        "tempo": 104,
        "vibrato": (5.0, 0.24),
        "lines": [
            "E4 D4 C4 D4 E4 E4",
            "D4 D4 D4 E4 G4 G4",
            "E4 D4 C4 D4 E4 E4",
            "D4 E4 F4 G4 E4 D4",
            "E4 E4 E4 D4 D4 D4 E4 D4",
            "D4 D4 D4 E4 E4 C4",
        ],
    },
    "row row row your boat": {
        "tempo": 112,
        "vibrato": (5.4, 0.2),
        "lines": [
            "C4 C4 C4 D4 E4",
            "E4 D4 E4 F4 G4",
            "E4 D4 C4 E4 D4 C4",
            "D4 C4 C4 C4 C4",
        ],
    },
    "itsy bitsy spider": {
        "tempo": 120,
        "vibrato": (5.6, 0.22),
        "lines": [
            "C4 D4 E4 C4 D4 E4 C4 D4 E4 F4 G4 A4 B4 C5",
            "G4 F4 E4 D4 C4 D4 E4 F4 G4 A4 B4",
        ],
    },
    "head shoulders knees and toes": {
        "tempo": 116,
        "vibrato": (5.2, 0.2),
        "lines": [
            "C4 D4 E4 F4 G4 G4",
            "G4 F4 E4",
            "C4 D4 E4 F4 G4 G4",
            "G4 F4 E4",
            "C4 D4 E4 F4 G4 A4 B4",
        ],
    },
    "baa baa black sheep": {
        "tempo": 108,
        "vibrato": (4.8, 0.24),
        "lines": [
            "C4 C4 D4 E4 D4 C4 A4 G4",
            "G4 A4 B4 A4 G4 E4 D4",
        ],
    },
    "hickory dickory dock": {
        "tempo": 118,
        "vibrato": (5.4, 0.2),
        "lines": [
            "C4 C4 D4 E4 D4",
            "E4 D4 C4 D4 E4 F4",
            "G4 G4 G4 G4",
            "C4 B4 A4 G4",
        ],
    },
    "pat a cake": {
        "tempo": 114,
        "vibrato": (5.2, 0.2),
        "lines": [
            "C4 D4 E4 F4 G4 G4 A4 G4 F4",
            "G4 F4 E4 D4 D4 E4 F4 E4 D4",
            "E4 F4 G4 E4 C4 C4 D4 E4 F4 G4",
            "G4 A4 G4 F4 E4 D4 C4 D4 E4 F4",
        ],
    },
    "london bridge": {
        "tempo": 100,
        "vibrato": (4.8, 0.26),
        "lines": [
            "G4 A4 G4 F4 E4 F4 G4 G4",
            "F4 E4 D4 E4 F4 G4",
            "G4 A4 G4 F4 E4 F4 G4 G4",
            "D4 E4 C4",
        ],
    },
    "three blind mice": {
        "tempo": 116,
        "vibrato": (5.4, 0.2),
        "lines": [
            "C4 D4 E4 F4 E4 D4 C4",
            "G4 A4 G4 F4 E4 D4 C4 D4 C4",
            "D4 E4 F4 E4 D4 C4",
        ],
    },
    "old macdonald had a farm": {
        "tempo": 122,
        "vibrato": (5.6, 0.22),
        "lines": [
            "G4 G4 G4 D4 E4 G4 G4",
            "G4 G4 D4 E4 G4",
            "E4 E4 E4 E4 E4 G4 C4 E4",
            "E4 E4 E4 E4 E4 G4 C4 C4 C4 C4",
        ],
    },
    "jingle bells": {
        "tempo": 128,
        "vibrato": (6.0, 0.18),
        "lines": [
            "E4 E4 E4 E4 E4 E4",
            "E4 E4 E4 E4 E4",
            "B4 B4 B4 B4 B4 B4 B4 B4 B4 B4 C4 C4 D4 E4 C4",
        ],
    },
    "auld lang syne": {
        "tempo": 84,
        "vibrato": (4.4, 0.3),
        "lines": [
            "C4 E4 G4 G4 E4 C4 D4 E4 E4",
            "E4 G4 G4 E4 C4 C4",
            "C4 E4 G4 G4 E4 C4 D4 E4 E4",
            "C4 C4 G4 G4 E4 C4",
        ],
    },
    "greensleeves": {
        "tempo": 76,
        "vibrato": (4.4, 0.3),
        "lines": [
            "A4 C5 B4 G4 A4 C5 A4 G4",
            "E4 G4 A4 G4 E4 G4 E4 D4 C4 E4",
        ],
    },
    "this old man": {
        "tempo": 120,
        "vibrato": (5.4, 0.2),
        "lines": [
            "G4 A4 G4 A4 G4 E4",
            "G4 A4 G4 A4 G4 E4 G4",
            "C5 C5 C5 G4 E4 D4 C4",
            "E4 E4 E4 D4 C4",
        ],
    },
}


def melody_for(name):
    """The stored note lines for a song, or None when it has no real melody."""
    entry = MELODIES.get(str(name or "").strip().casefold())
    return list(entry["lines"]) if entry else None

