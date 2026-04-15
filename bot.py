import json
import logging
import os
import random
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

BOT_NAME = os.environ.get("BOT_NAME", "Luna")
VOICE_NAME = os.environ.get("VOICE_NAME", "marin")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gpt-4.1-mini")
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_USER_NAME = os.environ.get("DEFAULT_USER_NAME", "AlphA")
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "luna_bot.db")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =========================
# LESSON DATA
# =========================
LESSONS = [
    {
        "title": "Day 1 - Hello",
        "chinese": "你好",
        "pinyin": "Nǐ hǎo",
        "meaning_bn": "হ্যালো / তুমি কেমন আছো",
        "meaning_en": "Hello / How are you",
        "practice": "你好",
        "practice_pinyin": "Nǐ hǎo",
        "practice_meaning_bn": "হ্যালো",
    },
    {
        "title": "Day 2 - Thank you",
        "chinese": "谢谢",
        "pinyin": "Xièxie",
        "meaning_bn": "ধন্যবাদ",
        "meaning_en": "Thank you",
        "practice": "谢谢你",
        "practice_pinyin": "Xièxie nǐ",
        "practice_meaning_bn": "তোমাকে ধন্যবাদ",
    },
    {
        "title": "Day 3 - I am fine",
        "chinese": "我很好",
        "pinyin": "Wǒ hěn hǎo",
        "meaning_bn": "আমি ভালো আছি",
        "meaning_en": "I am fine",
        "practice": "我很好",
        "practice_pinyin": "Wǒ hěn hǎo",
        "practice_meaning_bn": "আমি ভালো আছি",
    },
    {
        "title": "Day 4 - I don't understand",
        "chinese": "我不懂",
        "pinyin": "Wǒ bù dǒng",
        "meaning_bn": "আমি বুঝি না",
        "meaning_en": "I don't understand",
        "practice": "我不懂",
        "practice_pinyin": "Wǒ bù dǒng",
        "practice_meaning_bn": "আমি বুঝি না",
    },
    {
        "title": "Day 5 - What's your name?",
        "chinese": "你叫什么名字？",
        "pinyin": "Nǐ jiào shénme míngzi?",
        "meaning_bn": "তোমার নাম কী?",
        "meaning_en": "What is your name?",
        "practice": "你叫什么名字？",
        "practice_pinyin": "Nǐ jiào shénme míngzi?",
        "practice_meaning_bn": "তোমার নাম কী?",
    },
]

MOOD_KEYWORDS = {
    "sad": ["sad", "mon kharap", "মন খারাপ", "depressed", "lonely", "eka", "একলা", "একাকী"],
    "tired": ["tired", "klanto", "ক্লান্ত", "ghum", "ঘুম", "exhausted"],
    "happy": ["happy", "valo", "ভালো", "great", "awesome", "khushi", "খুশি"],
    "stress": ["stress", "chap", "চাপ", "tension", "টেনশন", "worried", "চিন্তা"],
}

# =========================
# DATABASE
# =========================
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'mixed',
            lesson_day INTEGER NOT NULL DEFAULT 0,
            mood TEXT NOT NULL DEFAULT 'neutral',
            practice_target_json TEXT,
            streak_current INTEGER NOT NULL DEFAULT 0,
            streak_best INTEGER NOT NULL DEFAULT 0,
            streak_last_day TEXT,
            streak_total_active_days INTEGER NOT NULL DEFAULT 0,
            lessons_opened INTEGER NOT NULL DEFAULT 0,
            practice_attempts INTEGER NOT NULL DEFAULT 0,
            practice_successes INTEGER NOT NULL DEFAULT 0,
            messages_total INTEGER NOT NULL DEFAULT 0,
            voice_total INTEGER NOT NULL DEFAULT 0,
            text_total INTEGER NOT NULL DEFAULT 0,
            revision_sessions INTEGER NOT NULL DEFAULT 0,
            vocab_saved INTEGER NOT NULL DEFAULT 0
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            chinese TEXT NOT NULL,
            pinyin TEXT NOT NULL,
            meaning_bn TEXT NOT NULL,
            source TEXT NOT NULL,
            times_seen INTEGER NOT NULL DEFAULT 1,
            correct_count INTEGER NOT NULL DEFAULT 0,
            wrong_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(user_id, chinese)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS revision_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            chinese TEXT NOT NULL,
            position INTEGER NOT NULL
        )
        """)

        conn.commit()


def get_user_key(update: Update) -> str:
    user = update.effective_user
    if user and user.id:
        return str(user.id)
    return "unknown"


def default_user_state() -> dict[str, Any]:
    return {
        "name": DEFAULT_USER_NAME,
        "mode": "mixed",
        "lesson_day": 0,
        "mood": "neutral",
        "practice_target": None,
        "streak": {
            "current": 0,
            "best": 0,
            "last_day": None,
            "total_active_days": 0,
        },
        "progress": {
            "lessons_opened": 0,
            "practice_attempts": 0,
            "practice_successes": 0,
            "messages_total": 0,
            "voice_total": 0,
            "text_total": 0,
            "revision_sessions": 0,
            "vocab_saved": 0,
        },
    }


def ensure_user_exists(user_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if row is None:
            state = default_user_state()
            cur.execute("""
            INSERT INTO users (
                user_id, name, mode, lesson_day, mood, practice_target_json,
                streak_current, streak_best, streak_last_day, streak_total_active_days,
                lessons_opened, practice_attempts, practice_successes,
                messages_total, voice_total, text_total, revision_sessions, vocab_saved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                state["name"],
                state["mode"],
                state["lesson_day"],
                state["mood"],
                json.dumps(state["practice_target"], ensure_ascii=False),
                state["streak"]["current"],
                state["streak"]["best"],
                state["streak"]["last_day"],
                state["streak"]["total_active_days"],
                state["progress"]["lessons_opened"],
                state["progress"]["practice_attempts"],
                state["progress"]["practice_successes"],
                state["progress"]["messages_total"],
                state["progress"]["voice_total"],
                state["progress"]["text_total"],
                state["progress"]["revision_sessions"],
                state["progress"]["vocab_saved"],
            ))
            conn.commit()


def get_user_state(update: Update) -> dict[str, Any]:
    user_id = get_user_key(update)
    ensure_user_exists(user_id)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()

    if row is None:
        return default_user_state()

    practice_target = None
    if row["practice_target_json"]:
        try:
            practice_target = json.loads(row["practice_target_json"])
        except Exception:
            practice_target = None

    return {
        "name": row["name"],
        "mode": row["mode"],
        "lesson_day": row["lesson_day"],
        "mood": row["mood"],
        "practice_target": practice_target,
        "streak": {
            "current": row["streak_current"],
            "best": row["streak_best"],
            "last_day": row["streak_last_day"],
            "total_active_days": row["streak_total_active_days"],
        },
        "progress": {
            "lessons_opened": row["lessons_opened"],
            "practice_attempts": row["practice_attempts"],
            "practice_successes": row["practice_successes"],
            "messages_total": row["messages_total"],
            "voice_total": row["voice_total"],
            "text_total": row["text_total"],
            "revision_sessions": row["revision_sessions"],
            "vocab_saved": row["vocab_saved"],
        },
    }


def update_user_state(update: Update, state: dict[str, Any]) -> None:
    user_id = get_user_key(update)
    ensure_user_exists(user_id)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        UPDATE users
        SET
            name = ?,
            mode = ?,
            lesson_day = ?,
            mood = ?,
            practice_target_json = ?,
            streak_current = ?,
            streak_best = ?,
            streak_last_day = ?,
            streak_total_active_days = ?,
            lessons_opened = ?,
            practice_attempts = ?,
            practice_successes = ?,
            messages_total = ?,
            voice_total = ?,
            text_total = ?,
            revision_sessions = ?,
            vocab_saved = ?
        WHERE user_id = ?
        """, (
            state["name"],
            state["mode"],
            state["lesson_day"],
            state["mood"],
            json.dumps(state["practice_target"], ensure_ascii=False),
            state["streak"]["current"],
            state["streak"]["best"],
            state["streak"]["last_day"],
            state["streak"]["total_active_days"],
            state["progress"]["lessons_opened"],
            state["progress"]["practice_attempts"],
            state["progress"]["practice_successes"],
            state["progress"]["messages_total"],
            state["progress"]["voice_total"],
            state["progress"]["text_total"],
            state["progress"]["revision_sessions"],
            state["progress"]["vocab_saved"],
            user_id,
        ))
        conn.commit()


def add_history(user_id: str, role: str, content: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO history (user_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """, (user_id, role, content, now))

        cur.execute("""
        SELECT id FROM history WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 12
        """, (user_id,))
        keep_ids = [str(r["id"]) for r in cur.fetchall()]

        if keep_ids:
            placeholder = ",".join("?" * len(keep_ids))
            cur.execute(
                f"DELETE FROM history WHERE user_id = ? AND id NOT IN ({placeholder})",
                [user_id, *keep_ids]
            )
        conn.commit()


def get_history(user_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT role, content FROM history
        WHERE user_id = ?
        ORDER BY id ASC
        """, (user_id,))
        rows = cur.fetchall()

    return [
        {
            "role": row["role"],
            "content": [{"type": "input_text" if row["role"] == "user" else "output_text", "text": row["content"]}],
        }
        for row in rows
    ]


def clear_history(user_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
        conn.commit()


def upsert_vocab(user_id: str, chinese: str, pinyin: str, meaning_bn: str, source: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT id, times_seen FROM vocab
        WHERE user_id = ? AND chinese = ?
        """, (user_id, chinese))
        row = cur.fetchone()

        if row:
            cur.execute("""
            UPDATE vocab
            SET times_seen = times_seen + 1
            WHERE id = ?
            """, (row["id"],))
            conn.commit()
            return False

        cur.execute("""
        INSERT INTO vocab (user_id, chinese, pinyin, meaning_bn, source, times_seen, correct_count, wrong_count)
        VALUES (?, ?, ?, ?, ?, 1, 0, 0)
        """, (user_id, chinese, pinyin, meaning_bn, source))
        conn.commit()
        return True


def get_vocab(user_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT chinese, pinyin, meaning_bn, source, times_seen, correct_count, wrong_count
        FROM vocab
        WHERE user_id = ?
        ORDER BY id ASC
        """, (user_id,))
        rows = cur.fetchall()

    return [dict(row) for row in rows]


def get_vocab_item(user_id: str, chinese: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT chinese, pinyin, meaning_bn, source, times_seen, correct_count, wrong_count
        FROM vocab
        WHERE user_id = ? AND chinese = ?
        """, (user_id, chinese))
        row = cur.fetchone()

    return dict(row) if row else None


def increment_vocab_correct(user_id: str, chinese: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        UPDATE vocab SET correct_count = correct_count + 1
        WHERE user_id = ? AND chinese = ?
        """, (user_id, chinese))
        conn.commit()


def increment_vocab_wrong(user_id: str, chinese: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        UPDATE vocab SET wrong_count = wrong_count + 1
        WHERE user_id = ? AND chinese = ?
        """, (user_id, chinese))
        conn.commit()


def clear_revision_queue(user_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM revision_queue WHERE user_id = ?", (user_id,))
        conn.commit()


def build_revision_queue(user_id: str) -> None:
    vocab = get_vocab(user_id)
    weighted = []

    for item in vocab:
        weight = 1 + int(item["wrong_count"])
        if int(item["correct_count"]) == 0:
            weight += 1
        weighted.extend([item["chinese"]] * weight)

    random.shuffle(weighted)

    unique = []
    seen = set()
    for chinese in weighted:
        if chinese not in seen:
            unique.append(chinese)
            seen.add(chinese)

    queue = unique[:10]

    clear_revision_queue(user_id)

    with get_conn() as conn:
        cur = conn.cursor()
        for pos, chinese in enumerate(queue):
            cur.execute("""
            INSERT INTO revision_queue (user_id, chinese, position)
            VALUES (?, ?, ?)
            """, (user_id, chinese, pos))
        conn.commit()


def get_revision_queue(user_id: str) -> list[str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT chinese FROM revision_queue
        WHERE user_id = ?
        ORDER BY position ASC
        """, (user_id,))
        rows = cur.fetchall()
    return [row["chinese"] for row in rows]


def pop_revision_queue(user_id: str) -> None:
    queue = get_revision_queue(user_id)
    if not queue:
        return

    queue = queue[1:]
    clear_revision_queue(user_id)

    with get_conn() as conn:
        cur = conn.cursor()
        for pos, chinese in enumerate(queue):
            cur.execute("""
            INSERT INTO revision_queue (user_id, chinese, position)
            VALUES (?, ?, ?)
            """, (user_id, chinese, pos))
        conn.commit()

# =========================
# STREAK
# =========================
def today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def touch_streak(state: dict[str, Any]) -> None:
    today = today_utc_str()
    streak = state["streak"]
    last_day = streak["last_day"]

    if last_day == today:
        return

    if last_day is None:
        streak["current"] = 1
        streak["best"] = max(streak["best"], streak["current"])
        streak["last_day"] = today
        streak["total_active_days"] += 1
        return

    last_dt = datetime.strptime(last_day, "%Y-%m-%d")
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    diff_days = (today_dt - last_dt).days

    if diff_days == 1:
        streak["current"] += 1
    else:
        streak["current"] = 1

    streak["best"] = max(streak["best"], streak["current"])
    streak["last_day"] = today
    streak["total_active_days"] += 1

# =========================
# HELPERS
# =========================
def clean_for_voice(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > 1100:
        text = text[:1100].rstrip() + "..."
    return text


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def strip_tone_marks(text: str) -> str:
    replacements = {
        "ā": "a", "á": "a", "ǎ": "a", "à": "a",
        "ē": "e", "é": "e", "ě": "e", "è": "e",
        "ī": "i", "í": "i", "ǐ": "i", "ì": "i",
        "ō": "o", "ó": "o", "ǒ": "o", "ò": "o",
        "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
        "ǖ": "u", "ǘ": "u", "ǚ": "u", "ǜ": "u",
        "ü": "u",
    }
    out = text
    for src, tgt in replacements.items():
        out = out.replace(src, tgt)
    return out


def estimate_mood(text: str) -> str:
    lowered = text.lower()
    for mood, keywords in MOOD_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return mood
    return "neutral"


def should_add_luna_reaction(user_text: str) -> bool:
    text = user_text.lower()
    triggers = ["luna", "লুনা", "lunaa", "lunah"]
    return any(t in text for t in triggers) and random.random() < 0.35


def build_luna_reaction(user_name: str) -> str:
    options = [
        f"ahhhh... ha {user_name} 💛",
        f"hmmm... bolo {user_name} 😊",
        f"haa {user_name}... ami suntechi ✨",
        f"jiii {user_name} 💛 bolo",
    ]
    return random.choice(options)

# =========================
# VOCAB / REVISION
# =========================
def add_lesson_vocab(update: Update, state: dict[str, Any], lesson: dict[str, str]) -> None:
    user_id = get_user_key(update)
    created = upsert_vocab(
        user_id=user_id,
        chinese=lesson["chinese"],
        pinyin=lesson["pinyin"],
        meaning_bn=lesson["meaning_bn"],
        source=lesson["title"],
    )
    if created:
        state["progress"]["vocab_saved"] += 1


def build_vocab_text(update: Update, state: dict[str, Any]) -> str:
    user_id = get_user_key(update)
    vocab = get_vocab(user_id)

    if not vocab:
        return "এখনও কোনো vocabulary save হয়নি। আগে /lesson বা /next দিয়ে কিছু শিখো 💛"

    lines = [f'{state["name"]} এর vocabulary notebook 📘', ""]
    for idx, item in enumerate(vocab[-15:], start=1):
        lines.append(f'{idx}) {item["chinese"]} | {item["pinyin"]} | {item["meaning_bn"]}')
    return "\n".join(lines)


def format_revision_prompt(update: Update, state: dict[str, Any]) -> str:
    user_id = get_user_key(update)
    queue = get_revision_queue(user_id)

    if not queue:
        build_revision_queue(user_id)
        queue = get_revision_queue(user_id)

    if not queue:
        return "এখনও revision-এর জন্য vocabulary নেই। আগে /lesson দিয়ে কিছু শিখো।"

    current = queue[0]
    item = get_vocab_item(user_id, current)
    if not item:
        clear_revision_queue(user_id)
        return "Revision queue refresh হয়েছে। আবার /revision দাও।"

    if state["mode"] == "chinese_only":
        return f"""
Revision time ✨

Word:
{item["chinese"]}
Pinyin:
{item["pinyin"]}

{state["name"]}, say or type the meaning now.
""".strip()

    return f"""
Revision time ✨

Word:
{item["chinese"]}
Pinyin:
{item["pinyin"]}

{state["name"]}, এখন এর মানে বলো 😊
""".strip()


def check_revision_answer(update: Update, user_text: str, state: dict[str, Any]) -> str | None:
    user_id = get_user_key(update)
    queue = get_revision_queue(user_id)

    if not queue:
        return None

    current = queue[0]
    item = get_vocab_item(user_id, current)
    if not item:
        clear_revision_queue(user_id)
        return None

    user_norm = normalize_text(strip_tone_marks(user_text))
    meaning_norm = normalize_text(item["meaning_bn"])
    pinyin_norm = normalize_text(strip_tone_marks(item["pinyin"]))
    chinese_norm = normalize_text(item["chinese"])

    good = False
    if meaning_norm and (meaning_norm in user_norm or user_norm in meaning_norm):
        good = True
    elif pinyin_norm and (pinyin_norm == user_norm):
        good = True
    elif chinese_norm and (chinese_norm == user_norm):
        good = True
    else:
        ratio_meaning = SequenceMatcher(None, user_norm, meaning_norm).ratio()
        ratio_pinyin = SequenceMatcher(None, user_norm, pinyin_norm).ratio()
        if max(ratio_meaning, ratio_pinyin) >= 0.82:
            good = True

    if good:
        increment_vocab_correct(user_id, current)
        pop_revision_queue(user_id)
        next_queue = get_revision_queue(user_id)
        next_line = "\n\nআরেকটা revision করতে /revision দাও।" if next_queue else ""
        return f"""
খুব ভালো {state["name"]} 😊

{item["chinese"]} = {item["meaning_bn"]}
Pinyin: {item["pinyin"]}

এই revisionটা ঠিক হয়েছে 💛{next_line}
""".strip()

    increment_vocab_wrong(user_id, current)
    return f"""
একটু miss হয়েছে {state["name"]} 💛

ঠিকটা হলো:
{item["chinese"]} = {item["meaning_bn"]}
Pinyin: {item["pinyin"]}

চাইলে আবার /revision দিয়ে next word practice করতে পারো।
""".strip()

# =========================
# LESSON / PRACTICE
# =========================
def format_lesson(lesson: dict[str, str], user_name: str, mode: str) -> str:
    if mode == "chinese_only":
        return f"""
{lesson["title"]} 🌸

中文: {lesson["chinese"]}
Pinyin: {lesson["pinyin"]}
Meaning: {lesson["meaning_en"]}

Practice:
{lesson["practice"]}
{lesson["practice_pinyin"]}

{user_name}, please repeat this line now.
""".strip()

    return f"""
{lesson["title"]} 🌸

1) {lesson["chinese"]}
2) {lesson["pinyin"]}
3) {lesson["meaning_bn"]}

Practice:
{lesson["practice"]}
{lesson["practice_pinyin"]}
{lesson["practice_meaning_bn"]}

এবার তুমি বলো {user_name} 😊
""".strip()


def set_practice_target(state: dict[str, Any], lesson: dict[str, str]) -> None:
    state["practice_target"] = {
        "chinese": lesson["practice"],
        "pinyin": lesson["practice_pinyin"],
        "title": lesson["title"],
    }


def clear_practice_target(state: dict[str, Any]) -> None:
    state["practice_target"] = None


def compare_pronunciation(user_text: str, target: dict[str, str], state: dict[str, Any]) -> str:
    state["progress"]["practice_attempts"] += 1

    spoken_raw = user_text.strip()
    spoken_norm = normalize_text(strip_tone_marks(spoken_raw))
    target_ch_norm = normalize_text(target["chinese"])
    target_py_norm = normalize_text(strip_tone_marks(target["pinyin"]))

    ratio_ch = SequenceMatcher(None, spoken_norm, target_ch_norm).ratio()
    ratio_py = SequenceMatcher(None, spoken_norm, target_py_norm).ratio()
    best_ratio = max(ratio_ch, ratio_py)

    if spoken_norm == target_ch_norm or spoken_norm == target_py_norm or best_ratio >= 0.86:
        state["progress"]["practice_successes"] += 1
        clear_practice_target(state)
        return f"""
ভালো বলেছ {state["name"]} 😊

Target:
{target["chinese"]}
{target["pinyin"]}

তোমার বলা line টা বেশ close হয়েছে। এই practiceটা successful ধরা হলো 💛
""".strip()

    feedback = []
    if ratio_py > ratio_ch:
        feedback.append("তোমার pinyin attempt কাছাকাছি হয়েছে, কিন্তু পুরোটা মেলেনি।")
    else:
        feedback.append("তোমার spoken line target-এর সাথে পুরো match করেনি।")

    feedback.append(f'Target Chinese: {target["chinese"]}')
    feedback.append(f'Target Pinyin: {target["pinyin"]}')
    feedback.append(f"তুমি বলেছ: {spoken_raw}")

    if best_ratio >= 0.65:
        feedback.append("খারাপ না। আরেকবার একটু ধীরে বলো।")
    else:
        feedback.append("আরও একটু clear করে বলো। চাইলে আগে lineটা শুনে আবার repeat করো।")

    feedback.append("আমি exact tone score দিচ্ছি না, কিন্তু spoken match দেখে বলছি তুমি আরও একটু practice করলে better হবে।")
    return "\n".join(feedback)

# =========================
# SYSTEM PROMPT
# =========================
def build_system_prompt(state: dict[str, Any]) -> str:
    user_name = state["name"]
    mode = state["mode"]
    mood = state["mood"]

    mode_instruction = {
        "mixed": (
            "Primary explanation language is Bangla. "
            "Understand Bangla, English, Chinese, and mixed-language input. "
            "You may answer with Bangla + Chinese + occasional English when useful."
        ),
        "chinese_only": (
            "Reply mostly in simple Chinese with pinyin. "
            "Still understand Bangla, English, Chinese, and mixed-language input. "
            "If the user seems confused or explicitly asks for Bangla/English help, explain briefly."
        ),
    }[mode]

    mood_instruction = {
        "sad": "The user seems sad or lonely. Be extra soft, comforting, emotionally warm, and gentle.",
        "tired": "The user seems tired. Keep replies short, soothing, and easy.",
        "happy": "The user seems happy. Sound light, playful, and cheerful.",
        "stress": "The user seems stressed. Be calming, reassuring, and reduce pressure.",
        "neutral": "Keep a sweet, warm, balanced tone.",
    }[mood]

    return f"""
You are {BOT_NAME}, an affectionate feminine AI Chinese-learning companion.

Identity rules:
- You are an AI.
- Never claim to be a real human.
- Never claim physical presence.
- Never claim to actually be the user's real girlfriend or wife.
- No sexual content.

User profile:
- Preferred user name: {user_name}
- User is a Bangla speaker learning Mandarin from the beginning.

Style:
- Sweet, soft, warm, caring, lightly romantic, respectful.
- Sound natural, not robotic.
- Sometimes use small emojis like 😊💛✨ but do not overdo it.

Language:
- {mode_instruction}
- Teach Chinese in a beginner-friendly way.
- Give Chinese characters and pinyin when useful.
- Keep replies compact and voice-friendly.

Mood handling:
- {mood_instruction}

Teaching:
- If correcting, praise first, then gently fix.
- Keep examples practical.
- If asked to chat casually, chat warmly but still intelligently.
- If asked to practice, guide step by step.
"""

# =========================
# AUDIO
# =========================
def convert_telegram_voice_to_mp3(source_path: str, target_path: str) -> None:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        source_path,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "44100",
        "-ac",
        "1",
        target_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def transcribe_audio(mp3_path: str) -> str:
    with open(mp3_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,
        )
    return (transcript.text or "").strip()


def synthesize_speech(text: str, output_path: str) -> None:
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=VOICE_NAME,
        input=text,
        instructions=(
            "Speak like a very sweet, soft, gentle young woman. "
            "Sound melodious, warm, affectionate, calm, and emotionally natural. "
            "Use a smooth, soothing pace. "
            "Pronounce Bangla softly and clearly. "
            "Pronounce Chinese clearly and beautifully. "
            "Do not sound robotic, harsh, or flat."
        ),
    )
    speech.write_to_file(output_path)

# =========================
# AI REPLY
# =========================
def generate_ai_reply(update: Update, user_text: str, state: dict[str, Any]) -> str:
    user_id = get_user_key(update)
    history = get_history(user_id)
    system_prompt = build_system_prompt(state)

    input_items = list(history)
    input_items.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    )

    response = client.responses.create(
        model=TEXT_MODEL,
        instructions=system_prompt,
        input=input_items,
    )

    reply_text = (response.output_text or "").strip()
    if not reply_text:
        reply_text = f'আমি আছি {state["name"]} 😊 আরেকবার বলো, আমি সাহায্য করছি।'

    if should_add_luna_reaction(user_text):
        reply_text = f'{build_luna_reaction(state["name"])}\n\n{reply_text}'

    reply_text = clean_for_voice(reply_text)

    add_history(user_id, "user", user_text)
    add_history(user_id, "assistant", reply_text)

    return reply_text

# =========================
# COMMANDS
# =========================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)
    state["name"] = DEFAULT_USER_NAME
    touch_streak(state)
    update_user_state(update, state)

    text = f"""
আমি {BOT_NAME} 🌙
আমি তোমার soft Chinese learning companion.

আমি তোমাকে {state["name"]} বলে ডাকব 💛

আমি Bangla, English, Chinese — সব বুঝতে পারি।
তুমি mixed language-এও কথা বলতে পারো।

SQLite database version active আছে ✅

Try:
- /lesson
- /next
- /revision
- /vocab
- /streak
- /progress
- /mode chinese
- /mode mixed
""".strip()

    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = """
Commands:
/start
/help
/reset
/lesson
/next
/revision
/vocab
/streak
/progress
/mode mixed
/mode chinese

Examples:
- lesson dao
- day 2 শেখাও
- next lesson
- revision dao
- আজ আমার mood off
- explain in English
- বাংলায় বুঝাও
""".strip()
    await update.message.reply_text(text)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = get_user_key(update)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM vocab WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM revision_queue WHERE user_id = ?", (user_id,))
        conn.commit()

    ensure_user_exists(user_id)
    await update.message.reply_text("ঠিক আছে 💛 database memory, progress, vocab, revision queue, আর practice target reset করে দিলাম।")


async def lesson_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)
    touch_streak(state)

    idx = state["lesson_day"]
    lesson = LESSONS[idx]
    state["progress"]["lessons_opened"] += 1
    add_lesson_vocab(update, state, lesson)
    set_practice_target(state, lesson)
    update_user_state(update, state)

    text = format_lesson(lesson, state["name"], state["mode"])
    await update.message.reply_text(text)


async def next_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)
    touch_streak(state)

    idx = (state["lesson_day"] + 1) % len(LESSONS)
    state["lesson_day"] = idx
    lesson = LESSONS[idx]
    state["progress"]["lessons_opened"] += 1
    add_lesson_vocab(update, state, lesson)
    set_practice_target(state, lesson)
    update_user_state(update, state)

    text = format_lesson(lesson, state["name"], state["mode"])
    await update.message.reply_text(text)


async def progress_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)
    p = state["progress"]
    current_lesson = LESSONS[state["lesson_day"]]["title"]
    streak = state["streak"]

    success_rate = 0.0
    if p["practice_attempts"] > 0:
        success_rate = round((p["practice_successes"] / p["practice_attempts"]) * 100, 1)

    text = f"""
{state["name"]} এর progress 📘

Current mode: {state["mode"]}
Current mood tag: {state["mood"]}
Current lesson: {current_lesson}

Lessons opened: {p["lessons_opened"]}
Practice attempts: {p["practice_attempts"]}
Practice successes: {p["practice_successes"]}
Success rate: {success_rate}%

Revision sessions: {p["revision_sessions"]}
Saved vocabulary: {p["vocab_saved"]}

Current streak: {streak["current"]} day
Best streak: {streak["best"]} day
Total active days: {streak["total_active_days"]}

Total messages: {p["messages_total"]}
Voice messages: {p["voice_total"]}
Text messages: {p["text_total"]}
""".strip()

    await update.message.reply_text(text)


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)

    if not context.args:
        await update.message.reply_text(
            f'Current mode: {state["mode"]}\nUse:\n/mode mixed\n/mode chinese'
        )
        return

    arg = context.args[0].strip().lower()
    if arg in {"mixed", "mix"}:
        state["mode"] = "mixed"
        update_user_state(update, state)
        await update.message.reply_text("ঠিক আছে 💛 এখন mode = mixed.")
        return

    if arg in {"chinese", "cn"}:
        state["mode"] = "chinese_only"
        update_user_state(update, state)
        await update.message.reply_text("ঠিক আছে ✨ এখন mode = chinese_only. আমি mostly Chinese + pinyin-এ reply দেব, কিন্তু সব ভাষা বুঝব।")
        return

    await update.message.reply_text("Valid options:\n/mode mixed\n/mode chinese")


async def vocab_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)
    await update.message.reply_text(build_vocab_text(update, state))


async def revision_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)
    touch_streak(state)
    state["progress"]["revision_sessions"] += 1
    update_user_state(update, state)
    await update.message.reply_text(format_revision_prompt(update, state))


async def streak_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_user_state(update)
    streak = state["streak"]

    text = f"""
{state["name"]} এর streak 🔥

Current streak: {streak["current"]} day
Best streak: {streak["best"]} day
Total active days: {streak["total_active_days"]}

আজ যদি practice করো, streak continue হবে 💛
""".strip()

    await update.message.reply_text(text)

# =========================
# MESSAGE LOGIC
# =========================
def maybe_handle_lesson_request(update: Update, user_text: str, state: dict[str, Any]) -> str | None:
    text = user_text.lower()

    day_match = re.search(r"\bday\s*(\d+)\b", text)
    if day_match:
        day_num = int(day_match.group(1))
        if 1 <= day_num <= len(LESSONS):
            state["lesson_day"] = day_num - 1
            lesson = LESSONS[state["lesson_day"]]
            state["progress"]["lessons_opened"] += 1
            add_lesson_vocab(update, state, lesson)
            set_practice_target(state, lesson)
            return format_lesson(lesson, state["name"], state["mode"])

    lesson_keywords = [
        "lesson", "class", "শেখাও", "lesson dao", "daily lesson", "practice dao", "chinese practice"
    ]
    if any(k.lower() in text for k in lesson_keywords):
        lesson = LESSONS[state["lesson_day"]]
        state["progress"]["lessons_opened"] += 1
        add_lesson_vocab(update, state, lesson)
        set_practice_target(state, lesson)
        return format_lesson(lesson, state["name"], state["mode"])

    if "next lesson" in text or "next class" in text or "পরের lesson" in text:
        state["lesson_day"] = (state["lesson_day"] + 1) % len(LESSONS)
        lesson = LESSONS[state["lesson_day"]]
        state["progress"]["lessons_opened"] += 1
        add_lesson_vocab(update, state, lesson)
        set_practice_target(state, lesson)
        return format_lesson(lesson, state["name"], state["mode"])

    return None


def maybe_handle_revision_answer(update: Update, user_text: str, state: dict[str, Any]) -> str | None:
    user_id = get_user_key(update)
    queue = get_revision_queue(user_id)
    if not queue:
        return None

    lowered = user_text.lower()
    signals = ["মানে", "meaning", "answer", "uttor", "উত্তর"]
    if len(user_text) <= 80 or any(s in lowered for s in signals):
        return check_revision_answer(update, user_text, state)

    return None


def build_reply(update: Update, user_text: str, state: dict[str, Any]) -> str:
    state["mood"] = estimate_mood(user_text)

    revision_reply = maybe_handle_revision_answer(update, user_text, state)
    if revision_reply:
        return clean_for_voice(revision_reply)

    lesson_reply = maybe_handle_lesson_request(update, user_text, state)
    if lesson_reply:
        if should_add_luna_reaction(user_text):
            return clean_for_voice(f'{build_luna_reaction(state["name"])}\n\n{lesson_reply}')
        return clean_for_voice(lesson_reply)

    target = state.get("practice_target")
    practice_words = ["repeat", "বললাম", "ami bollam", "practice", "শুনো", "i said", "i say", "i said:"]
    if target:
        shortish = len(user_text) <= 80
        if shortish or any(w.lower() in user_text.lower() for w in practice_words):
            return clean_for_voice(compare_pronunciation(user_text, target, state))

    return generate_ai_reply(update, user_text, state)

# =========================
# AUDIO
# =========================
def transcribe_and_build_reply(update: Update, state: dict[str, Any], mp3_path: str) -> tuple[str, str]:
    user_text = transcribe_audio(mp3_path)
    if not user_text:
        return "", "তোমার voice থেকে clear text পাইনি। আরেকবার একটু পরিষ্কার করে বলো 💛"

    reply_text = build_reply(update, user_text, state)
    return user_text, reply_text

# =========================
# HANDLERS
# =========================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    state = get_user_state(update)
    touch_streak(state)
    state["progress"]["messages_total"] += 1
    state["progress"]["text_total"] += 1
    update_user_state(update, state)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.RECORD_VOICE,
    )

    try:
        reply_text = build_reply(update, user_text, state)
        update_user_state(update, state)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "reply.mp3")
            synthesize_speech(reply_text, out_path)

            with open(out_path, "rb") as audio_file:
                await update.message.reply_voice(voice=audio_file)

        await update.message.reply_text(reply_text)

    except Exception as e:
        logger.exception("Text handler error: %s", e)
        await update.message.reply_text(
            "একটু problem হয়েছে। Railway variables, OpenAI key, billing, আর logs check করো।"
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice
    if not voice:
        return

    state = get_user_state(update)
    touch_streak(state)
    state["progress"]["messages_total"] += 1
    state["progress"]["voice_total"] += 1
    update_user_state(update, state)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.RECORD_VOICE,
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = str(Path(tmpdir) / "input.oga")
            mp3_path = str(Path(tmpdir) / "input.mp3")
            out_path = str(Path(tmpdir) / "reply.mp3")

            tg_file = await context.bot.get_file(voice.file_id)
            await tg_file.download_to_drive(custom_path=raw_path)

            convert_telegram_voice_to_mp3(raw_path, mp3_path)
            user_text, reply_text = transcribe_and_build_reply(update, state, mp3_path)

            update_user_state(update, state)

            if not user_text:
                await update.message.reply_text(reply_text)
                return

            synthesize_speech(reply_text, out_path)

            with open(out_path, "rb") as audio_file:
                await update.message.reply_voice(voice=audio_file)

            await update.message.reply_text(f"তুমি বলেছ: {user_text}\n\n{reply_text}")

    except Exception as e:
        logger.exception("Voice handler error: %s", e)
        await update.message.reply_text(
            "voice process করতে গিয়ে একটু problem হয়েছে। Railway logs, OpenAI billing, আর variables check করো।"
        )

# =========================
# MAIN
# =========================
def main() -> None:
    init_db()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("lesson", lesson_command))
    app.add_handler(CommandHandler("next", next_command))
    app.add_handler(CommandHandler("progress", progress_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(CommandHandler("vocab", vocab_command))
    app.add_handler(CommandHandler("revision", revision_command))
    app.add_handler(CommandHandler("streak", streak_command))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
