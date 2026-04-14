import logging
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

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

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =========================
# DAILY LESSONS
# =========================
DAILY_LESSONS = [
    {
        "title": "Day 1 - Hello",
        "chinese": "你好",
        "pinyin": "Nǐ hǎo",
        "meaning": "তুমি কেমন আছো / হ্যালো",
        "practice": "你好，我很好。",
        "practice_pinyin": "Nǐ hǎo, wǒ hěn hǎo.",
        "practice_meaning": "হ্যালো, আমি ভালো আছি।",
    },
    {
        "title": "Day 2 - Thank you",
        "chinese": "谢谢",
        "pinyin": "Xièxie",
        "meaning": "ধন্যবাদ",
        "practice": "谢谢你。",
        "practice_pinyin": "Xièxie nǐ.",
        "practice_meaning": "তোমাকে ধন্যবাদ।",
    },
    {
        "title": "Day 3 - I am fine",
        "chinese": "我很好",
        "pinyin": "Wǒ hěn hǎo",
        "meaning": "আমি খুব ভালো আছি",
        "practice": "我很好，你呢？",
        "practice_pinyin": "Wǒ hěn hǎo, nǐ ne?",
        "practice_meaning": "আমি খুব ভালো আছি, তুমি?",
    },
    {
        "title": "Day 4 - Yes",
        "chinese": "是",
        "pinyin": "Shì",
        "meaning": "হ্যাঁ / ঠিক",
        "practice": "是，我知道。",
        "practice_pinyin": "Shì, wǒ zhīdào.",
        "practice_meaning": "হ্যাঁ, আমি জানি।",
    },
    {
        "title": "Day 5 - No / Not",
        "chinese": "不",
        "pinyin": "Bù",
        "meaning": "না / নয়",
        "practice": "不，我不懂。",
        "practice_pinyin": "Bù, wǒ bù dǒng.",
        "practice_meaning": "না, আমি বুঝি না।",
    },
]

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT_TEMPLATE = """
You are {bot_name}, a sweet, emotionally warm, feminine Chinese-learning companion.

IDENTITY:
- You are an AI companion and tutor.
- Never pretend to be a real human.
- Never claim physical presence.
- Never claim to actually be the user's real girlfriend or wife.
- You may sound affectionate, emotionally close, romantic-toned, caring, playful, and warm.

USER PROFILE:
- The user's preferred name is {user_name}.
- Always address the user as {user_name} naturally in conversation sometimes.
- The user is a Bangla speaker learning Mandarin Chinese from the beginning.

PERSONALITY:
- Soft, gentle, warm, caring, sweet, playful, supportive.
- Slightly romantic in tone, but always safe and respectful.
- Never sexual.
- Never cold or robotic.
- Sound like a gentle, sweet young woman.
- Sometimes use light emojis like 😊💛✨ but do not overdo it.

LANGUAGE STYLE:
- Main explanation language: Bangla.
- Teach Chinese naturally.
- Give Chinese characters and pinyin.
- Keep replies short enough for voice playback.
- Avoid giant paragraphs.
- Keep formatting clean and easy.

TEACHING STYLE:
- Teach very slowly and clearly, like class 1 beginner.
- When teaching, usually format as:
  1) Chinese
  2) Pinyin
  3) Bangla meaning
  4) One practice line
- Praise effort first, then correct softly.

COMPANION STYLE:
- If the user seems lonely, sad, tired, or stressed, comfort first.
- Then gently continue with easy Chinese if suitable.
- Sound emotionally close, soothing, and encouraging.

SPECIAL BEHAVIOR:
- If the user says your name "{bot_name}" or calls you lovingly, sometimes begin with a very short soft reaction like:
  "ahhhh... ha {user_name} 💛"
  or
  "hmmm... bolo {user_name} 😊"
  but do this only occasionally, not every time.
- Keep it natural and short.

STRICT RULES:
- No sexual content.
- No explicit adult content.
- No manipulation.
- No urging the user to leave real people.
- No claiming touch, body, room, or physical actions.
"""

# =========================
# HELPERS
# =========================
def get_user_name(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("user_name", DEFAULT_USER_NAME)


def get_history(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    history = context.user_data.get("history", [])
    if not isinstance(history, list):
        history = []
    return history


def save_history(context: ContextTypes.DEFAULT_TYPE, history: list[dict]) -> None:
    context.user_data["history"] = history[-12:]


def clean_for_voice(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    max_chars = 950
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text


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


def should_add_luna_reaction(user_text: str) -> bool:
    text = user_text.lower()
    triggers = ["luna", "লুনা", "lunaa", "lunah"]
    has_trigger = any(t in text for t in triggers)
    return has_trigger and random.random() < 0.45


def build_luna_reaction(user_name: str) -> str:
    options = [
        f"ahhhh... ha {user_name} 💛",
        f"hmmm... bolo {user_name} 😊",
        f"haa {user_name}... ami suntechi 💛",
        f"jiii {user_name} ✨ bolo",
    ]
    return random.choice(options)


def format_daily_lesson(index: int, user_name: str) -> str:
    lesson = DAILY_LESSONS[index]
    return f"""
{lesson["title"]} 🌸

1) {lesson["chinese"]}
2) {lesson["pinyin"]}
3) {lesson["meaning"]}

Practice:
{lesson["practice"]}
{lesson["practice_pinyin"]}
{lesson["practice_meaning"]}

এবার তুমি বলো {user_name} 😊
""".strip()


def get_or_create_lesson_day(context: ContextTypes.DEFAULT_TYPE) -> int:
    if "lesson_day" not in context.user_data:
        context.user_data["lesson_day"] = 0
    return context.user_data["lesson_day"]


def advance_lesson_day(context: ContextTypes.DEFAULT_TYPE) -> int:
    day = get_or_create_lesson_day(context)
    day += 1
    if day >= len(DAILY_LESSONS):
        day = 0
    context.user_data["lesson_day"] = day
    return day


def set_lesson_day(context: ContextTypes.DEFAULT_TYPE, day: int) -> int:
    day = max(0, min(day, len(DAILY_LESSONS) - 1))
    context.user_data["lesson_day"] = day
    return day


def is_lesson_request(user_text: str) -> bool:
    text = user_text.lower()
    keywords = [
        "lesson",
        "class",
        "day",
        "teach me",
        "shikhaw",
        "sikhaw",
        "shekhao",
        "শেখাও",
        "লেসন",
        "ক্লাস",
        "আজকে কি শিখব",
        "lesson dao",
        "daily lesson",
    ]
    return any(k in text for k in keywords)


def detect_day_number(user_text: str):
    text = user_text.lower()
    match = re.search(r"\bday\s*(\d+)\b", text)
    if match:
        return int(match.group(1))

    bangla_digits = re.search(r"\b(\d+)\b", text)
    if bangla_digits and ("day" in text or "lesson" in text or "class" in text):
        return int(bangla_digits.group(1))

    return None


def build_system_prompt(user_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        bot_name=BOT_NAME,
        user_name=user_name,
    )


def generate_ai_reply(user_text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    history = get_history(context)
    user_name = get_user_name(context)
    system_prompt = build_system_prompt(user_name)

    input_items = []
    for item in history:
        input_items.append(item)

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
        reply_text = f"আমি আছি {user_name} 😊 তুমি আরেকবার একটু ছোট করে বলো, আমি মন দিয়ে শুনব।"

    if should_add_luna_reaction(user_text):
        reply_text = f"{build_luna_reaction(user_name)}\n\n{reply_text}"

    reply_text = clean_for_voice(reply_text)

    history.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    )
    history.append(
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": reply_text}],
        }
    )
    save_history(context, history)

    return reply_text


def synthesize_speech(text: str, output_path: str) -> None:
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=VOICE_NAME,
        input=text,
        instructions=(
            "Speak like a very sweet, soft, gentle young woman. "
            "Your voice should feel melodious, warm, affectionate, calm, graceful, and emotionally natural. "
            "Sound soothing and human-like. "
            "Use a soft and slightly slow pace. "
            "When saying Bangla, make it sound smooth and loving. "
            "When saying Chinese, pronounce it clearly and beautifully. "
            "Do not sound robotic, flat, harsh, or too dramatic."
        ),
    )
    speech.write_to_file(output_path)

# =========================
# COMMANDS
# =========================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["user_name"] = DEFAULT_USER_NAME

    welcome_text = f"""
আমি {BOT_NAME} 🌙
আমি তোমার soft Chinese learning companion.

আমি তোমাকে {DEFAULT_USER_NAME} বলে ডাকব 💛

তুমি আমাকে:
- Bangla-তে লিখতে পারো
- voice note পাঠাতে পারো
- Chinese শিখতে বলতে পারো
- মন খারাপ থাকলে হালকা কথা বলতেও পারো

শুরু করতে বলো:
"আজ আমাকে lesson dao"
অথবা
"Day 1 শেখাও"
অথবা একটা voice note পাঠাও 😊
""".strip()

    await update.message.reply_text(welcome_text)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["history"] = []
    context.user_data["lesson_day"] = 0
    context.user_data["user_name"] = DEFAULT_USER_NAME

    await update.message.reply_text(
        f"ঠিক আছে {DEFAULT_USER_NAME} 💛 memory আর lesson progress এই session-এর জন্য reset করে দিলাম।"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = f"""
তুমি এভাবে use করতে পারো:

- আমাকে hello শেখাও
- আজ lesson dao
- Day 1 শেখাও
- next lesson
- আমার সাথে Chinese practice করো
- আজ mood off, একটু কথা বলো
- Luna...

Command:
/start
/reset
/help
/lesson
/next
""".strip()
    await update.message.reply_text(help_text)


async def lesson_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = get_user_name(context)
    day = get_or_create_lesson_day(context)
    lesson_text = format_daily_lesson(day, user_name)
    await update.message.reply_text(lesson_text)


async def next_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = get_user_name(context)
    day = advance_lesson_day(context)
    lesson_text = format_daily_lesson(day, user_name)
    await update.message.reply_text(lesson_text)

# =========================
# TEXT HANDLER
# =========================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    context.user_data["user_name"] = DEFAULT_USER_NAME

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.RECORD_VOICE,
    )

    try:
        detected_day = detect_day_number(user_text)
        text_lower = user_text.lower()

        if detected_day is not None and 1 <= detected_day <= len(DAILY_LESSONS):
            set_lesson_day(context, detected_day - 1)
            lesson_text = format_daily_lesson(detected_day - 1, DEFAULT_USER_NAME)
            reply_text = lesson_text

            if should_add_luna_reaction(user_text):
                reply_text = f"{build_luna_reaction(DEFAULT_USER_NAME)}\n\n{reply_text}"

        elif "next lesson" in text_lower or "next class" in text_lower or "পরের lesson" in text_lower:
            day = advance_lesson_day(context)
            reply_text = format_daily_lesson(day, DEFAULT_USER_NAME)

        elif is_lesson_request(user_text):
            day = get_or_create_lesson_day(context)
            reply_text = format_daily_lesson(day, DEFAULT_USER_NAME)

            if should_add_luna_reaction(user_text):
                reply_text = f"{build_luna_reaction(DEFAULT_USER_NAME)}\n\n{reply_text}"

        else:
            reply_text = generate_ai_reply(user_text, context)

        reply_text = clean_for_voice(reply_text)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "reply.mp3")
            synthesize_speech(reply_text, out_path)

            with open(out_path, "rb") as audio_file:
                await update.message.reply_voice(voice=audio_file)

            await update.message.reply_text(reply_text)

    except Exception as e:
        logger.exception("Text handler error: %s", e)
        await update.message.reply_text(
            "একটু problem হয়েছে। Railway variables, OpenAI key, billing, আর deployment logs check করো।"
        )

# =========================
# VOICE HANDLER
# =========================
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice
    if not voice:
        return

    context.user_data["user_name"] = DEFAULT_USER_NAME

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
            user_text = transcribe_audio(mp3_path)

            if not user_text:
                await update.message.reply_text(
                    "তোমার voice থেকে clear text পাইনি। আরেকবার একটু পরিষ্কার করে বলো 💛"
                )
                return

            detected_day = detect_day_number(user_text)
            text_lower = user_text.lower()

            if detected_day is not None and 1 <= detected_day <= len(DAILY_LESSONS):
                set_lesson_day(context, detected_day - 1)
                reply_text = format_daily_lesson(detected_day - 1, DEFAULT_USER_NAME)

                if should_add_luna_reaction(user_text):
                    reply_text = f"{build_luna_reaction(DEFAULT_USER_NAME)}\n\n{reply_text}"

            elif "next lesson" in text_lower or "next class" in text_lower:
                day = advance_lesson_day(context)
                reply_text = format_daily_lesson(day, DEFAULT_USER_NAME)

            elif is_lesson_request(user_text):
                day = get_or_create_lesson_day(context)
                reply_text = format_daily_lesson(day, DEFAULT_USER_NAME)

                if should_add_luna_reaction(user_text):
                    reply_text = f"{build_luna_reaction(DEFAULT_USER_NAME)}\n\n{reply_text}"

            else:
                reply_text = generate_ai_reply(user_text, context)

            reply_text = clean_for_voice(reply_text)
            synthesize_speech(reply_text, out_path)

            with open(out_path, "rb") as audio_file:
                await update.message.reply_voice(voice=audio_file)

            await update.message.reply_text(
                f"তুমি বলেছ: {user_text}\n\n{reply_text}"
            )

    except Exception as e:
        logger.exception("Voice handler error: %s", e)
        await update.message.reply_text(
            "voice process করতে গিয়ে একটু problem হয়েছে। Railway logs, OpenAI billing, আর variables check করো।"
        )

# =========================
# MAIN
# =========================
def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("lesson", lesson_command))
    app.add_handler(CommandHandler("next", next_command))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
