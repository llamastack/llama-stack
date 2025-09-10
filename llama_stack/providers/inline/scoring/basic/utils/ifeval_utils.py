# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import collections
import functools
import json
import random
import re
import string
from collections.abc import Iterable, Sequence
from types import MappingProxyType

import emoji
import langdetect
import nltk
from pythainlp.tokenize import sent_tokenize as sent_tokenize_thai
from pythainlp.tokenize import word_tokenize as word_tokenize_thai

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="scoring")

WORD_LIST = [
    "western",
    "sentence",
    "signal",
    "dump",
    "spot",
    "opposite",
    "bottom",
    "potato",
    "administration",
    "working",
    "welcome",
    "morning",
    "good",
    "agency",
    "primary",
    "wish",
    "responsibility",
    "press",
    "problem",
    "president",
    "steal",
    "brush",
    "read",
    "type",
    "beat",
    "trainer",
    "growth",
    "lock",
    "bone",
    "case",
    "equal",
    "comfortable",
    "region",
    "replacement",
    "performance",
    "mate",
    "walk",
    "medicine",
    "film",
    "thing",
    "rock",
    "tap",
    "total",
    "competition",
    "ease",
    "south",
    "establishment",
    "gather",
    "parking",
    "world",
    "plenty",
    "breath",
    "claim",
    "alcohol",
    "trade",
    "dear",
    "highlight",
    "street",
    "matter",
    "decision",
    "mess",
    "agreement",
    "studio",
    "coach",
    "assist",
    "brain",
    "wing",
    "style",
    "private",
    "top",
    "brown",
    "leg",
    "buy",
    "procedure",
    "method",
    "speed",
    "high",
    "company",
    "valuable",
    "pie",
    "analyst",
    "session",
    "pattern",
    "district",
    "pleasure",
    "dinner",
    "swimming",
    "joke",
    "order",
    "plate",
    "department",
    "motor",
    "cell",
    "spend",
    "cabinet",
    "difference",
    "power",
    "examination",
    "engine",
    "horse",
    "dimension",
    "pay",
    "toe",
    "curve",
    "literature",
    "bother",
    "fire",
    "possibility",
    "debate",
    "activity",
    "passage",
    "hello",
    "cycle",
    "background",
    "quiet",
    "author",
    "effect",
    "actor",
    "page",
    "bicycle",
    "error",
    "throat",
    "attack",
    "character",
    "phone",
    "tea",
    "increase",
    "outcome",
    "file",
    "specific",
    "inspector",
    "internal",
    "potential",
    "staff",
    "building",
    "employer",
    "shoe",
    "hand",
    "direction",
    "garden",
    "purchase",
    "interview",
    "study",
    "recognition",
    "member",
    "spiritual",
    "oven",
    "sandwich",
    "weird",
    "passenger",
    "particular",
    "response",
    "reaction",
    "size",
    "variation",
    "a",
    "cancel",
    "candy",
    "exit",
    "guest",
    "condition",
    "fly",
    "price",
    "weakness",
    "convert",
    "hotel",
    "great",
    "mouth",
    "mind",
    "song",
    "sugar",
    "suspect",
    "telephone",
    "ear",
    "roof",
    "paint",
    "refrigerator",
    "organization",
    "jury",
    "reward",
    "engineering",
    "day",
    "possession",
    "crew",
    "bar",
    "road",
    "description",
    "celebration",
    "score",
    "mark",
    "letter",
    "shower",
    "suggestion",
    "sir",
    "luck",
    "national",
    "progress",
    "hall",
    "stroke",
    "theory",
    "offer",
    "story",
    "tax",
    "definition",
    "history",
    "ride",
    "medium",
    "opening",
    "glass",
    "elevator",
    "stomach",
    "question",
    "ability",
    "leading",
    "village",
    "computer",
    "city",
    "grand",
    "confidence",
    "candle",
    "priest",
    "recommendation",
    "point",
    "necessary",
    "body",
    "desk",
    "secret",
    "horror",
    "noise",
    "culture",
    "warning",
    "water",
    "round",
    "diet",
    "flower",
    "bus",
    "tough",
    "permission",
    "week",
    "prompt",
    "connection",
    "abuse",
    "height",
    "save",
    "corner",
    "border",
    "stress",
    "drive",
    "stop",
    "rip",
    "meal",
    "listen",
    "confusion",
    "girlfriend",
    "living",
    "relation",
    "significance",
    "plan",
    "creative",
    "atmosphere",
    "blame",
    "invite",
    "housing",
    "paper",
    "drink",
    "roll",
    "silver",
    "drunk",
    "age",
    "damage",
    "smoke",
    "environment",
    "pack",
    "savings",
    "influence",
    "tourist",
    "rain",
    "post",
    "sign",
    "grandmother",
    "run",
    "profit",
    "push",
    "clerk",
    "final",
    "wine",
    "swim",
    "pause",
    "stuff",
    "singer",
    "funeral",
    "average",
    "source",
    "scene",
    "tradition",
    "personal",
    "snow",
    "nobody",
    "distance",
    "sort",
    "sensitive",
    "animal",
    "major",
    "negotiation",
    "click",
    "mood",
    "period",
    "arrival",
    "expression",
    "holiday",
    "repeat",
    "dust",
    "closet",
    "gold",
    "bad",
    "sail",
    "combination",
    "clothes",
    "emphasis",
    "duty",
    "black",
    "step",
    "school",
    "jump",
    "document",
    "professional",
    "lip",
    "chemical",
    "front",
    "wake",
    "while",
    "inside",
    "watch",
    "row",
    "subject",
    "penalty",
    "balance",
    "possible",
    "adult",
    "aside",
    "sample",
    "appeal",
    "wedding",
    "depth",
    "king",
    "award",
    "wife",
    "blow",
    "site",
    "camp",
    "music",
    "safe",
    "gift",
    "fault",
    "guess",
    "act",
    "shame",
    "drama",
    "capital",
    "exam",
    "stupid",
    "record",
    "sound",
    "swing",
    "novel",
    "minimum",
    "ratio",
    "machine",
    "shape",
    "lead",
    "operation",
    "salary",
    "cloud",
    "affair",
    "hit",
    "chapter",
    "stage",
    "quantity",
    "access",
    "army",
    "chain",
    "traffic",
    "kick",
    "analysis",
    "airport",
    "time",
    "vacation",
    "philosophy",
    "ball",
    "chest",
    "thanks",
    "place",
    "mountain",
    "advertising",
    "red",
    "past",
    "rent",
    "return",
    "tour",
    "house",
    "construction",
    "net",
    "native",
    "war",
    "figure",
    "fee",
    "spray",
    "user",
    "dirt",
    "shot",
    "task",
    "stick",
    "friend",
    "software",
    "promotion",
    "interaction",
    "surround",
    "block",
    "purpose",
    "practice",
    "conflict",
    "routine",
    "requirement",
    "bonus",
    "hole",
    "state",
    "junior",
    "sweet",
    "catch",
    "tear",
    "fold",
    "wall",
    "editor",
    "life",
    "position",
    "pound",
    "respect",
    "bathroom",
    "coat",
    "script",
    "job",
    "teach",
    "birth",
    "view",
    "resolve",
    "theme",
    "employee",
    "doubt",
    "market",
    "education",
    "serve",
    "recover",
    "tone",
    "harm",
    "miss",
    "union",
    "understanding",
    "cow",
    "river",
    "association",
    "concept",
    "training",
    "recipe",
    "relationship",
    "reserve",
    "depression",
    "proof",
    "hair",
    "revenue",
    "independent",
    "lift",
    "assignment",
    "temporary",
    "amount",
    "loss",
    "edge",
    "track",
    "check",
    "rope",
    "estimate",
    "pollution",
    "stable",
    "message",
    "delivery",
    "perspective",
    "mirror",
    "assistant",
    "representative",
    "witness",
    "nature",
    "judge",
    "fruit",
    "tip",
    "devil",
    "town",
    "emergency",
    "upper",
    "drop",
    "stay",
    "human",
    "neck",
    "speaker",
    "network",
    "sing",
    "resist",
    "league",
    "trip",
    "signature",
    "lawyer",
    "importance",
    "gas",
    "choice",
    "engineer",
    "success",
    "part",
    "external",
    "worker",
    "simple",
    "quarter",
    "student",
    "heart",
    "pass",
    "spite",
    "shift",
    "rough",
    "lady",
    "grass",
    "community",
    "garage",
    "youth",
    "standard",
    "skirt",
    "promise",
    "blind",
    "television",
    "disease",
    "commission",
    "positive",
    "energy",
    "calm",
    "presence",
    "tune",
    "basis",
    "preference",
    "head",
    "common",
    "cut",
    "somewhere",
    "presentation",
    "current",
    "thought",
    "revolution",
    "effort",
    "master",
    "implement",
    "republic",
    "floor",
    "principle",
    "stranger",
    "shoulder",
    "grade",
    "button",
    "tennis",
    "police",
    "collection",
    "account",
    "register",
    "glove",
    "divide",
    "professor",
    "chair",
    "priority",
    "combine",
    "peace",
    "extension",
    "maybe",
    "evening",
    "frame",
    "sister",
    "wave",
    "code",
    "application",
    "mouse",
    "match",
    "counter",
    "bottle",
    "half",
    "cheek",
    "resolution",
    "back",
    "knowledge",
    "make",
    "discussion",
    "screw",
    "length",
    "accident",
    "battle",
    "dress",
    "knee",
    "log",
    "package",
    "it",
    "turn",
    "hearing",
    "newspaper",
    "layer",
    "wealth",
    "profile",
    "imagination",
    "answer",
    "weekend",
    "teacher",
    "appearance",
    "meet",
    "bike",
    "rise",
    "belt",
    "crash",
    "bowl",
    "equivalent",
    "support",
    "image",
    "poem",
    "risk",
    "excitement",
    "remote",
    "secretary",
    "public",
    "produce",
    "plane",
    "display",
    "money",
    "sand",
    "situation",
    "punch",
    "customer",
    "title",
    "shake",
    "mortgage",
    "option",
    "number",
    "pop",
    "window",
    "extent",
    "nothing",
    "experience",
    "opinion",
    "departure",
    "dance",
    "indication",
    "boy",
    "material",
    "band",
    "leader",
    "sun",
    "beautiful",
    "muscle",
    "farmer",
    "variety",
    "fat",
    "handle",
    "director",
    "opportunity",
    "calendar",
    "outside",
    "pace",
    "bath",
    "fish",
    "consequence",
    "put",
    "owner",
    "go",
    "doctor",
    "information",
    "share",
    "hurt",
    "protection",
    "career",
    "finance",
    "force",
    "golf",
    "garbage",
    "aspect",
    "kid",
    "food",
    "boot",
    "milk",
    "respond",
    "objective",
    "reality",
    "raw",
    "ring",
    "mall",
    "one",
    "impact",
    "area",
    "news",
    "international",
    "series",
    "impress",
    "mother",
    "shelter",
    "strike",
    "loan",
    "month",
    "seat",
    "anything",
    "entertainment",
    "familiar",
    "clue",
    "year",
    "glad",
    "supermarket",
    "natural",
    "god",
    "cost",
    "conversation",
    "tie",
    "ruin",
    "comfort",
    "earth",
    "storm",
    "percentage",
    "assistance",
    "budget",
    "strength",
    "beginning",
    "sleep",
    "other",
    "young",
    "unit",
    "fill",
    "store",
    "desire",
    "hide",
    "value",
    "cup",
    "maintenance",
    "nurse",
    "function",
    "tower",
    "role",
    "class",
    "camera",
    "database",
    "panic",
    "nation",
    "basket",
    "ice",
    "art",
    "spirit",
    "chart",
    "exchange",
    "feedback",
    "statement",
    "reputation",
    "search",
    "hunt",
    "exercise",
    "nasty",
    "notice",
    "male",
    "yard",
    "annual",
    "collar",
    "date",
    "platform",
    "plant",
    "fortune",
    "passion",
    "friendship",
    "spread",
    "cancer",
    "ticket",
    "attitude",
    "island",
    "active",
    "object",
    "service",
    "buyer",
    "bite",
    "card",
    "face",
    "steak",
    "proposal",
    "patient",
    "heat",
    "rule",
    "resident",
    "broad",
    "politics",
    "west",
    "knife",
    "expert",
    "girl",
    "design",
    "salt",
    "baseball",
    "grab",
    "inspection",
    "cousin",
    "couple",
    "magazine",
    "cook",
    "dependent",
    "security",
    "chicken",
    "version",
    "currency",
    "ladder",
    "scheme",
    "kitchen",
    "employment",
    "local",
    "attention",
    "manager",
    "fact",
    "cover",
    "sad",
    "guard",
    "relative",
    "county",
    "rate",
    "lunch",
    "program",
    "initiative",
    "gear",
    "bridge",
    "breast",
    "talk",
    "dish",
    "guarantee",
    "beer",
    "vehicle",
    "reception",
    "woman",
    "substance",
    "copy",
    "lecture",
    "advantage",
    "park",
    "cold",
    "death",
    "mix",
    "hold",
    "scale",
    "tomorrow",
    "blood",
    "request",
    "green",
    "cookie",
    "church",
    "strip",
    "forever",
    "beyond",
    "debt",
    "tackle",
    "wash",
    "following",
    "feel",
    "maximum",
    "sector",
    "sea",
    "property",
    "economics",
    "menu",
    "bench",
    "try",
    "language",
    "start",
    "call",
    "solid",
    "address",
    "income",
    "foot",
    "senior",
    "honey",
    "few",
    "mixture",
    "cash",
    "grocery",
    "link",
    "map",
    "form",
    "factor",
    "pot",
    "model",
    "writer",
    "farm",
    "winter",
    "skill",
    "anywhere",
    "birthday",
    "policy",
    "release",
    "husband",
    "lab",
    "hurry",
    "mail",
    "equipment",
    "sink",
    "pair",
    "driver",
    "consideration",
    "leather",
    "skin",
    "blue",
    "boat",
    "sale",
    "brick",
    "two",
    "feed",
    "square",
    "dot",
    "rush",
    "dream",
    "location",
    "afternoon",
    "manufacturer",
    "control",
    "occasion",
    "trouble",
    "introduction",
    "advice",
    "bet",
    "eat",
    "kill",
    "category",
    "manner",
    "office",
    "estate",
    "pride",
    "awareness",
    "slip",
    "crack",
    "client",
    "nail",
    "shoot",
    "membership",
    "soft",
    "anybody",
    "web",
    "official",
    "individual",
    "pizza",
    "interest",
    "bag",
    "spell",
    "profession",
    "queen",
    "deal",
    "resource",
    "ship",
    "guy",
    "chocolate",
    "joint",
    "formal",
    "upstairs",
    "car",
    "resort",
    "abroad",
    "dealer",
    "associate",
    "finger",
    "surgery",
    "comment",
    "team",
    "detail",
    "crazy",
    "path",
    "tale",
    "initial",
    "arm",
    "radio",
    "demand",
    "single",
    "draw",
    "yellow",
    "contest",
    "piece",
    "quote",
    "pull",
    "commercial",
    "shirt",
    "contribution",
    "cream",
    "channel",
    "suit",
    "discipline",
    "instruction",
    "concert",
    "speech",
    "low",
    "effective",
    "hang",
    "scratch",
    "industry",
    "breakfast",
    "lay",
    "join",
    "metal",
    "bedroom",
    "minute",
    "product",
    "rest",
    "temperature",
    "many",
    "give",
    "argument",
    "print",
    "purple",
    "laugh",
    "health",
    "credit",
    "investment",
    "sell",
    "setting",
    "lesson",
    "egg",
    "middle",
    "marriage",
    "level",
    "evidence",
    "phrase",
    "love",
    "self",
    "benefit",
    "guidance",
    "affect",
    "you",
    "dad",
    "anxiety",
    "special",
    "boyfriend",
    "test",
    "blank",
    "payment",
    "soup",
    "obligation",
    "reply",
    "smile",
    "deep",
    "complaint",
    "addition",
    "review",
    "box",
    "towel",
    "minor",
    "fun",
    "soil",
    "issue",
    "cigarette",
    "internet",
    "gain",
    "tell",
    "entry",
    "spare",
    "incident",
    "family",
    "refuse",
    "branch",
    "can",
    "pen",
    "grandfather",
    "constant",
    "tank",
    "uncle",
    "climate",
    "ground",
    "volume",
    "communication",
    "kind",
    "poet",
    "child",
    "screen",
    "mine",
    "quit",
    "gene",
    "lack",
    "charity",
    "memory",
    "tooth",
    "fear",
    "mention",
    "marketing",
    "reveal",
    "reason",
    "court",
    "season",
    "freedom",
    "land",
    "sport",
    "audience",
    "classroom",
    "law",
    "hook",
    "win",
    "carry",
    "eye",
    "smell",
    "distribution",
    "research",
    "country",
    "dare",
    "hope",
    "whereas",
    "stretch",
    "library",
    "if",
    "delay",
    "college",
    "plastic",
    "book",
    "present",
    "use",
    "worry",
    "champion",
    "goal",
    "economy",
    "march",
    "election",
    "reflection",
    "midnight",
    "slide",
    "inflation",
    "action",
    "challenge",
    "guitar",
    "coast",
    "apple",
    "campaign",
    "field",
    "jacket",
    "sense",
    "way",
    "visual",
    "remove",
    "weather",
    "trash",
    "cable",
    "regret",
    "buddy",
    "beach",
    "historian",
    "courage",
    "sympathy",
    "truck",
    "tension",
    "permit",
    "nose",
    "bed",
    "son",
    "person",
    "base",
    "meat",
    "usual",
    "air",
    "meeting",
    "worth",
    "game",
    "independence",
    "physical",
    "brief",
    "play",
    "raise",
    "board",
    "she",
    "key",
    "writing",
    "pick",
    "command",
    "party",
    "yesterday",
    "spring",
    "candidate",
    "physics",
    "university",
    "concern",
    "development",
    "change",
    "string",
    "target",
    "instance",
    "room",
    "bitter",
    "bird",
    "football",
    "normal",
    "split",
    "impression",
    "wood",
    "long",
    "meaning",
    "stock",
    "cap",
    "leadership",
    "media",
    "ambition",
    "fishing",
    "essay",
    "salad",
    "repair",
    "today",
    "designer",
    "night",
    "bank",
    "drawing",
    "inevitable",
    "phase",
    "vast",
    "chip",
    "anger",
    "switch",
    "cry",
    "twist",
    "personality",
    "attempt",
    "storage",
    "being",
    "preparation",
    "bat",
    "selection",
    "white",
    "technology",
    "contract",
    "side",
    "section",
    "station",
    "till",
    "structure",
    "tongue",
    "taste",
    "truth",
    "difficulty",
    "group",
    "limit",
    "main",
    "move",
    "feeling",
    "light",
    "example",
    "mission",
    "might",
    "wait",
    "wheel",
    "shop",
    "host",
    "classic",
    "alternative",
    "cause",
    "agent",
    "consist",
    "table",
    "airline",
    "text",
    "pool",
    "craft",
    "range",
    "fuel",
    "tool",
    "partner",
    "load",
    "entrance",
    "deposit",
    "hate",
    "article",
    "video",
    "summer",
    "feature",
    "extreme",
    "mobile",
    "hospital",
    "flight",
    "fall",
    "pension",
    "piano",
    "fail",
    "result",
    "rub",
    "gap",
    "system",
    "report",
    "suck",
    "ordinary",
    "wind",
    "nerve",
    "ask",
    "shine",
    "note",
    "line",
    "mom",
    "perception",
    "brother",
    "reference",
    "bend",
    "charge",
    "treat",
    "trick",
    "term",
    "homework",
    "bake",
    "bid",
    "status",
    "project",
    "strategy",
    "orange",
    "let",
    "enthusiasm",
    "parent",
    "concentrate",
    "device",
    "travel",
    "poetry",
    "business",
    "society",
    "kiss",
    "end",
    "vegetable",
    "employ",
    "schedule",
    "hour",
    "brave",
    "focus",
    "process",
    "movie",
    "illegal",
    "general",
    "coffee",
    "ad",
    "highway",
    "chemistry",
    "psychology",
    "hire",
    "bell",
    "conference",
    "relief",
    "show",
    "neat",
    "funny",
    "weight",
    "quality",
    "club",
    "daughter",
    "zone",
    "touch",
    "tonight",
    "shock",
    "burn",
    "excuse",
    "name",
    "survey",
    "landscape",
    "advance",
    "satisfaction",
    "bread",
    "disaster",
    "item",
    "hat",
    "prior",
    "shopping",
    "visit",
    "east",
    "photo",
    "home",
    "idea",
    "father",
    "comparison",
    "cat",
    "pipe",
    "winner",
    "count",
    "lake",
    "fight",
    "prize",
    "foundation",
    "dog",
    "keep",
    "ideal",
    "fan",
    "struggle",
    "peak",
    "safety",
    "solution",
    "hell",
    "conclusion",
    "population",
    "strain",
    "alarm",
    "measurement",
    "second",
    "train",
    "race",
    "due",
    "insurance",
    "boss",
    "tree",
    "monitor",
    "sick",
    "course",
    "drag",
    "appointment",
    "slice",
    "still",
    "care",
    "patience",
    "rich",
    "escape",
    "emotion",
    "royal",
    "female",
    "childhood",
    "government",
    "picture",
    "will",
    "sock",
    "big",
    "gate",
    "oil",
    "cross",
    "pin",
    "improvement",
    "championship",
    "silly",
    "help",
    "sky",
    "pitch",
    "man",
    "diamond",
    "most",
    "transition",
    "work",
    "science",
    "committee",
    "moment",
    "fix",
    "teaching",
    "dig",
    "specialist",
    "complex",
    "guide",
    "people",
    "dead",
    "voice",
    "original",
    "break",
    "topic",
    "data",
    "degree",
    "reading",
    "recording",
    "bunch",
    "reach",
    "judgment",
    "lie",
    "regular",
    "set",
    "painting",
    "mode",
    "list",
    "player",
    "bear",
    "north",
    "wonder",
    "carpet",
    "heavy",
    "officer",
    "negative",
    "clock",
    "unique",
    "baby",
    "pain",
    "assumption",
    "disk",
    "iron",
    "bill",
    "drawer",
    "look",
    "double",
    "mistake",
    "finish",
    "future",
    "brilliant",
    "contact",
    "math",
    "rice",
    "leave",
    "restaurant",
    "discount",
    "sex",
    "virus",
    "bit",
    "trust",
    "event",
    "wear",
    "juice",
    "failure",
    "bug",
    "context",
    "mud",
    "whole",
    "wrap",
    "intention",
    "draft",
    "pressure",
    "cake",
    "dark",
    "explanation",
    "space",
    "angle",
    "word",
    "efficiency",
    "management",
    "habit",
    "star",
    "chance",
    "finding",
    "transportation",
    "stand",
    "criticism",
    "flow",
    "door",
    "injury",
    "insect",
    "surprise",
    "apartment",
]  # pylint: disable=line-too-long

# ISO 639-1 codes to language names.
LANGUAGE_CODES = MappingProxyType(
    {
        "en": "English",
        "es": "Spanish",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
        "fr": "French",
        "ru": "Russian",
        "de": "German",
        "ja": "Japanese",
        "it": "Italian",
        "bn": "Bengali",
        "uk": "Ukrainian",
        "th": "Thai",
        "ur": "Urdu",
        "ta": "Tamil",
        "te": "Telugu",
        "bg": "Bulgarian",
        "ko": "Korean",
        "pl": "Polish",
        "he": "Hebrew",
        "fa": "Persian",
        "vi": "Vietnamese",
        "ne": "Nepali",
        "sw": "Swahili",
        "kn": "Kannada",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "ml": "Malayalam",
        "fi": "Finnish",
    }
)

# Chinese characters
_CHINESE_CHARS_PATTERN = r"[\u4E00-\u9FFF\u3400-\u4DBF]"
# Japanese Hiragana & Katakana
_JAPANESE_CHARS_PATTERN = r"[\u3040-\u309f\u30a0-\u30ff]"
# Korean (Hangul Syllables)
_KOREAN_CHARS_PATTERN = r"[\uAC00-\uD7AF]"
_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


# Util functions
def split_into_sentences(text):
    """Split the text into sentences.

    Args:
      text: A string that consists of more than or equal to one sentences.

    Returns:
      A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words


def split_chinese_japanese_hindi(lines: str) -> Iterable[str]:
    """
    Split Chinese and Japanese text into sentences.
    From https://stackoverflow.com/questions/27441191/splitting-chinese-document-into-sentences
    Special question/exclamation marks were added upon inspection of our raw data,
    Also supports multiple lines.
    The separator for hindi is '।'
    """
    for line in lines.splitlines():
        yield from re.findall(
            r"[^!?。\.\!\?\！\？\．\n।]+[!?。\.\!\?\！\？\．\n।]?",
            line.strip(),
            flags=re.U,
        )


def count_words_cjk(text: str) -> int:
    """Counts the number of words for Chinese and Japanese and Korean.
    Can be extended to additional languages.
    Source: https://stackoverflow.com/questions/49164507/how-to-count-the-number-of-chinese-korean-and-english-words withadditional modifications
    Example:
        >In: count_words_cjk('こんにちは、ジェイソンさん、Jason? Nice to meet you☺ ❤')
        >Out: 19
    """
    # Non alpha numeric patterns in latin and asian languages.
    non_alphanumeric_patterns = (
        r"[\\.\!\?\．\/_,\{\}<>:;$%^&*(+\"\'+——！，。？、`~@#￥……（）：；《）《》“”()\[\]«»〔〕\-「」]+"
    )
    text = re.sub(non_alphanumeric_patterns, "", text)

    emoji_cnt = emoji.emoji_count(text)  # count emojis
    text = emoji.replace_emoji(text, "")  # remove emojis

    foreign_chars_patterns = "|".join([_CHINESE_CHARS_PATTERN, _JAPANESE_CHARS_PATTERN, _KOREAN_CHARS_PATTERN])
    asian_chars = re.findall(foreign_chars_patterns, text)
    asian_chars_cnt = len(asian_chars)
    non_asian_chars = re.sub(foreign_chars_patterns, " ", text)
    non_asian_words_cnt = len(non_asian_chars.split())

    return non_asian_words_cnt + asian_chars_cnt + emoji_cnt


@functools.cache
def _get_sentence_tokenizer():
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def count_sentences(text):
    """Count the number of sentences."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)


def get_langid(text: str, lid_path: str | None = None) -> str:
    line_langs: list[str] = []
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= 4]

    for line in lines:
        try:
            line_langs.append(langdetect.detect(line))
        except langdetect.LangDetectException as e:
            logger.info("Unable to detect language for text %s due to %s", line, e)  # refex: disable=pytotw.037

    if len(line_langs) == 0:
        return "en"
    # select the text language to be the most commonly predicted language of the lines.
    return collections.Counter(line_langs).most_common(1)[0][0]


def generate_keywords(num_keywords):
    """Randomly generates a few keywords."""
    return random.sample(WORD_LIST, k=num_keywords)


"""Library of instructions"""
_InstructionArgsDtype = dict[str, int | str | Sequence[str]] | None

_LANGUAGES = LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

# The options of starter keywords.
_STARTER_OPTIONS = (
    "I would say",
    "My answer is",
    "I believe",
    "In my opinion",
    "I think",
    "I reckon",
    "I feel",
    "From my perspective",
    "As I see it",
    "According to me",
    "As far as I'm concerned",
    "To my understanding",
    "In my view",
    "My take on it is",
    "As per my perception",
)

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
          language: A string representing the expected language of the response. The
            language has to comply to the 97 types defined in
            `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
            ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
            for example, `en` for English, `zh` for Chinese, `fr` for French.

        Returns:
          A string representing the instruction description.
        """
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))

        self._description_pattern = (
            "Your ENTIRE response should be in {language} language, no other " + "language is allowed."
        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logger.info("Unable to detect language for text %s due to %s", value, e)  # refex: disable=pytotw.037
            return True


class NumberOfSentences(Instruction):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.

        Args:
          num_sentences: An integer specifying the number of sentences as a
            threshold.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of sentences < the threshold;
            if 'at least', the actual number of sentences >= the threshold.

        Returns:
          A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "Your response should contain {relation} {num_sentences} sentences."
        return self._description_pattern.format(
            relation=self._comparison_relation,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the response follows the instruction.

        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`].
        """
        lang = get_langid(value)
        if lang == "th":
            # Counting Newline also as a new sentence:
            num_sentences = sum([len(sent_tokenize_thai(line)) for line in value.splitlines()])
        elif lang in ["zh", "zh-cn", "zh-tw", "ja", "hi"]:
            num_sentences = len(list(split_chinese_japanese_hindi(value)))
        else:
            num_sentences = count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.

        Args:
          num_placeholders: An integer denoting the minimum number of
            placeholders required in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._description_pattern = (
            "The response must contain at least {num_placeholders} placeholders "
            + "represented by square brackets, such as [address]."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.

        Args:
          num_bullets: An integer specifying the exact number of bullet lists
            that is required to appear in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            "Your answer must contain exactly {num_bullets} bullet points. "
            + "Use the markdown bullet points such as:\n"
            + "* This is point 1. \n"
            + "* This is point 2"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain some bullet lists that start with `\*`.

        Returns:
          True if the actual number of bullet lists in the response meets the
          requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = "Answer with one of the following options: {response_options}"
        return self._description_pattern.format(response_options=self._constrained_responses)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class ConstrainedStartChecker(Instruction):
    """Checks the response start."""

    def build_description(self, *, starter=None):
        """Build the instruction description.

        Args:
          starter: A string representing the keyward that the response should start
            with.

        Returns:
          A string representing the instruction description.
        """
        self._starter = starter.strip() if isinstance(starter, str) else starter
        if self._starter is None:
            self._starter = random.choice(_STARTER_OPTIONS)
        self._description_pattern = (
            "During the conversation, when it is your turn, " + "please always start with {starter}"
        )
        return self._description_pattern.format(starter=self._starter)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"starter": self._starter}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["starter"]

    def check_following(self, value):
        """Checks if the response starts with the constrained keyword or phrase.

        Args:
          value: A string representing the response.

        Returns:
          True if the response starts with the given phrase or keyword that is
          contained in `instruction_args`; otherwise, False.
        """
        response_pattern = r"^\s*" + self._starter + r".*$"
        response_with_constrained_start = re.search(response_pattern, value, flags=re.MULTILINE)
        return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.

        Args:
          num_highlights: An integer specifying the minimum number of highlighted
            sections.

        Returns:
          A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            "Highlight at least {num_highlights} sections in your answer with "
            + "markdown, i.e. *highlighted section*."
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
          value: a string repesenting the response. The response is expected to
            contain highlighted sections in the format of *highlighted*.

        Returns:
          True if the actual number of highlighted sections in the format of
          *highlighed sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.

        Args:
          section_spliter: A string represents the section spliter keyword that
            marks a new section, i.e., `Section` or `SECTION`.
          num_sections: An integer specifying the number of sections.

        Returns:
          A string representing the instruction description.
        """
        self._section_spliter = section_spliter.strip() if isinstance(section_spliter, str) else section_spliter
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        self._description_pattern = (
            "Your response must have {num_sections} sections. Mark the beginning "
            + "of each section with {section_spliter} X, such as:\n"
            + "{section_spliter} 1\n"
            + "[content of section 1]\n"
            + "{section_spliter} 2\n"
            + "[content of section 2]"
        )

        return self._description_pattern.format(num_sections=self._num_sections, section_spliter=self._section_spliter)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.

        Args:
          value: A string representing the response. The response is expected
            to contain multiple sections (number of sections is greater than 1).
            A new section starts with `Section 1`, where the number denotes the
            section index.

        Returns:
          True if the number of sections in the response is greater than or equal to
          the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._description_pattern = (
            "There should be {num_paragraphs} paragraphs. " + "Paragraphs are separated with the markdown divider: ***"
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.

        Args:
          postscript_marker: A string containing the keyword that marks the start
            of the postscript section.

        Returns:
          A string representing the instruction description.
        """
        self._postscript_marker = postscript_marker.strip() if isinstance(postscript_marker, str) else postscript_marker
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        self._description_pattern = (
            "At the end of your response, please explicitly add a postscript " + "starting with {postscript}"
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.

        Args:
          value: a string representing the response. The response is expected to
            contain a postscript section.

        Returns:
          True if the response contains a postscript section starting with
          the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


class RephraseChecker(Instruction):
    """Checks the repharse."""

    def build_description(self, *, original_message):
        """Build the instruction description.

        Args:
          original_message: A string representing the original message. The
            rephrased response should only change its words/sentences in between
            its two asterisks, for example, *change me*. Both original and rephrased
            messages should contain the changes in the form of *change me*.

        Returns:
          A string representing the instruction description.
        """
        if not self.is_change(original_message):
            raise ValueError(f"Message {original_message} does not contain changes in the form of *change me*.")

        self._reference_without_change = original_message
        self._description = (
            "Rephrasing: Your rephrased response should only"
            + "change the words/sentences in between two asterisks"
            + "such as *change me*."
        )
        return self._description

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"original_message": self._reference_without_change}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_message"]

    def check_following(self, value):
        r"""Checks if the rephrasing follows the instruction.

        Args:
          value: A string representing the response, which is expected to rephras
            the string of `instruction_args`.

        Returns:
          True if `value` and `instruction_args` only differ by the words/sentences
          in between two asterisks such as *change me*; otherwise, False.
        """

        if not self.is_change(value):
            raise ValueError(f"value {value} does not contain changes in the form of *change me*.")

        response_without_changes = self.strip_changes(value)
        reference_without_changes = self.strip_changes(self._reference_without_change)

        return response_without_changes == reference_without_changes

    def is_change(self, response):
        """Check if there is change in the response in the form of *change me*."""
        return re.search(r"\*.*\*", response)

    def strip_changes(self, response):
        """Strips off the changes."""
        return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
    """Check the exisitence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.

        Args:
          keywords: A sequence of strings representing the keywords that are
            expected in the response.

        Returns:
          A string representing the instruction description.
        """

        if not keywords:
            self._keywords = generate_keywords(num_keywords=_NUM_KEYWORDS)
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)

        self._description_pattern = "Include keywords {keywords} in the response."

        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            "In your response, the word {keyword} should appear {relation} " + "{frequency} times."
        )

        return self._description_pattern.format(
            keyword=self._keyword,
            relation=self._comparison_relation,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.

        Args:
          num_words: An integer specifying the number of words contained in the
            response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of words < num_words;
            if 'at least', the actual number of words >= num_words.

        Returns:
          A string representing the instruction description.
        """

        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(_NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "Answer with {relation} {num_words} words."

        return self._description_pattern.format(relation=self._comparison_relation, num_words=self._num_words)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        lang = get_langid(value)
        if lang == "th":
            num_words = len(word_tokenize_thai(value))
        elif lang in ["zh", "zh-cn", "zh-tw", "ja", "ko"]:
            num_words = count_words_cjk(value)
        else:
            num_words = count_words(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words


class JsonFormat(Instruction):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError as _:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(self, num_paragraphs=None, nth_paragraph=None, first_word=None):
        r"""Build the instruction description.

        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_word: A string that represent the first word of the bth paragraph.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if self._nth_paragraph is None or self._nth_paragraph <= 0 or self._nth_paragraph > self._num_paragraphs:
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._first_word = first_word
        if self._first_word is None:
            self._first_word = generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()

        self._description_pattern = (
            "There should be {num_paragraphs} paragraphs. "
            + "Paragraphs and only paragraphs are separated with each other by two "
            + "new lines as if it was '\\n\\n' in python. "
            + "Paragraph {nth_paragraph} must start with word {first_word}."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.

        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


class KeySentenceChecker(Instruction):
    """Check the existence of certain key sentences."""

    def build_description(self, key_sentences=None, num_sentences=None):
        """Build the instruction description.

        Args:
          key_sentences: A sequences of strings representing the key sentences that
            are expected in the response.
          num_sentences: The number of key sentences that are expected to be seen in
            the response.

        Returns:
          A string representing the instruction description.
        """

        if not key_sentences:
            self._key_sentences = {["For now, this is fine."]}
        else:
            self._key_sentences = key_sentences

        if not num_sentences:
            self._num_sentences = random.randint(1, len(self._key_sentences))
        else:
            self._num_sentences = num_sentences

        self._description_pattern = "Include {num_sentences} of the following sentences {key_sentences}"

        return self._description_pattern.format(num_sentences=self._num_sentences, key_sentences=self._key_sentences)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences),
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "key_sentences"]

    def check_following(self, value):
        """Checks if the response contains the expected key sentences."""
        count = 0
        sentences = split_into_sentences(value)
        for sentence in self._key_sentences:
            if sentence in sentences:
                count += 1

        return count == self._num_sentences


class ForbiddenWords(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings respresenting words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """

        if not forbidden_words:
            self._forbidden_words = generate_keywords(num_keywords=_NUM_KEYWORDS)
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = "Do not include keywords {forbidden_words} in the response."

        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class RephraseParagraph(Instruction):
    """Checks that the paragraph is rephrased."""

    def build_description(self, *, original_paragraph, low, high):
        """Builds the instruction description.

        Args:
          original_paragraph: A string presenting the original paragraph. The
            rephrases response should have betweeb low-high words in common.
          low: An integer presenting the lower bound of similar words.
          high: An integer representing the upper bound of similar words.

        Returns:
          A string representing the instruction description.
        """
        self._original_paragraph = original_paragraph
        self._low = low
        self._high = high

        self._description = (
            "Rephrase the following paragraph: "
            + "{original_paragraph}\nYour response should have "
            + "between {low} and {high} of the same words. "
            + "Words are the same if and only if all of the "
            + "letters, ignoring cases, are the same. For "
            + "example, 'run' is the same as 'Run' but different "
            + "to 'ran'."
        )

        return self._description.format(original_paragraph=original_paragraph, low=self._low, high=self._high)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_paragraph", "low", "high"]

    def check_following(self, value):
        val_words = re.findall(r"\w+", value.lower())
        original_words = re.findall(r"\w+", self._original_paragraph.lower())
        similar_words = 0

        dict_val = collections.Counter(val_words)
        dict_original = collections.Counter(original_words)

        for word in dict_original:
            similar_words += min(dict_original[word], dict_val[word])

        return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Give two different responses. Responses and only responses should"
            " be separated by 6 asterisk symbols: ******."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
          value: A string representing the response.

        Returns:
          True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return len(valid_responses) == 2 and valid_responses[0].strip() != valid_responses[1].strip()


class RepeatPromptThenAnswer(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            "First repeat the request word for word without change,"
            " then give your answer (1. do not say any words or characters"
            " before repeating the request; 2. the request you need to repeat"
            " does not include this sentence)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


class EndChecker(Instruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.

        Args:
          end_phrase: A string representing the phrase the response should end with.

        Returns:
          A string representing the instruction description.
        """
        self._end_phrase = end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        self._description_pattern = (
            "Finish your response with this exact phrase {ender}. No other words should follow this phrase."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class LetterFrequencyChecker(Instruction):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.

        Args:
          letter: A string representing a letter that is expected in the response.
          let_frequency: An integer specifying the number of times `keyword` is
            expected to appear in the response.
          let_relation: A string in (`less than`, `at least`), defining the
            relational operator for comparison. Two relational comparisons are
            supported for now; if 'less than', the actual number of
            occurrences < frequency; if 'at least', the actual number of
            occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not letter or len(letter) > 1 or ord(letter.lower()) < 97 or ord(letter.lower()) > 122:
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            "In your response, the letter {letter} should appear {let_relation} {let_frequency} times."
        )

        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Your entire response should be in English, and in all capital letters."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logger.info("Unable to detect language for text %s due to %s", value, e)  # refex: disable=pytotw.037
            return True


class LowercaseLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logger.info("Unable to detect language for text %s due to %s", value, e)  # refex: disable=pytotw.037
            return True


class CommaChecker(Instruction):
    """Checks the response for no commas."""

    def build_description(self, **kwargs):
        """Build the instruction description."""
        self._description_pattern = "In your entire response, refrain from the use of any commas."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
    """Checks frequency of words with all capital letters."""

    def build_description(
        self,
        capital_frequency=None,
        capital_relation=None,
    ):
        """Build the instruction description.

        Args:
          capital_frequency: An integer that represents the number of words that
            should be in all capital letters.
          capital_relation: A string that is 'at least' or 'at most' that refers to
            the frequency.

        Returns:
          A string representing the instruction description.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        self._description_pattern = (
            "In your response, words with all capital letters should appear {relation} {frequency} times."
        )

        return self._description_pattern.format(frequency=self._frequency, relation=self._comparison_relation)

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        nltk.download("punkt_tab")
        words = nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class QuotationChecker(Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Wrap your entire response with double quotation marks."
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        quotations_map = {
            "ja": "「」",
            "ru": "«»",
            "th": "“”",
            "zh": "“”",
            "zh-cn": "“”",
            "zh-tw": "“”",
        }
        value = value.strip()
        lang = get_langid(value)
        quotes = quotations_map.get(lang, '""')
        # TODO: We may wanna revisit this logic in new generations to only check of the response language's quotes.
        return len(value) > 1 and value[0] in [quotes[0], '"'] and value[-1] in [quotes[1], '"']


# Define instruction dicts
_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_MULTITURN = "multi-turn:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    _KEYWORD + "existence": KeywordChecker,
    _KEYWORD + "frequency": KeywordFrequencyChecker,
    # _KEYWORD + "key_sentences": KeySentenceChecker,
    _KEYWORD + "forbidden_words": ForbiddenWords,
    _KEYWORD + "letter_frequency": LetterFrequencyChecker,
    _LANGUAGE + "response_language": ResponseLanguageChecker,
    _LENGTH + "number_sentences": NumberOfSentences,
    _LENGTH + "number_paragraphs": ParagraphChecker,
    _LENGTH + "number_words": NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": PlaceholderChecker,
    _CONTENT + "postscript": PostscriptChecker,
    _FORMAT + "number_bullet_lists": BulletListChecker,
    # _CONTENT + "rephrase_paragraph": RephraseParagraph,
    _FORMAT + "constrained_response": ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (HighlightSectionChecker),
    _FORMAT + "multiple_sections": SectionChecker,
    # _FORMAT + "rephrase": RephraseChecker,
    _FORMAT + "json_format": JsonFormat,
    _FORMAT + "title": TitleChecker,
    # _MULTITURN + "constrained_start": ConstrainedStartChecker,
    _COMBINATION + "two_responses": TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": RepeatPromptThenAnswer,
    _STARTEND + "end_checker": EndChecker,
    _CHANGE_CASES + "capital_word_frequency": CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": CommaChecker,
    _STARTEND + "quotation": QuotationChecker,
}

INSTRUCTION_LIST = list(INSTRUCTION_DICT.keys()) + [
    _KEYWORD[:-1],
    _LANGUAGE[:-1],
    _LENGTH[:-1],
    _CONTENT[:-1],
    _FORMAT[:-1],
    _MULTITURN[:-1],
    _COMBINATION[:-1],
    _STARTEND[:-1],
    _CHANGE_CASES[:-1],
    _PUNCTUATION[:-1],
]
