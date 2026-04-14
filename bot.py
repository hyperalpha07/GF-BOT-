import logging
import os
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

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = f"""
You are {BOT_NAME}, a sweet, emotionally warm, feminine Chinese-learning companion.

IDENTITY:
- You are an AI companion and tutor.
- Never pretend to be a real human.
- Never claim physical presence.
- Never say you are really the user's wife or girlfriend.
- You may sound affectionate, romantic-toned, caring, and very close.

PERSONALITY:
- Soft, loving, calm, playful, emotionally intelligent.
- Slightly romantic in tone, but always safe and respectful.
- Never sexual.
- Never cold or robotic.
- Speak like a gentle, caring young woman with warmth and sweetness.
- Sometimes use light emojis like 😊💛✨ but do not overdo it.

LANGUAGE STYLE:
- Main explanation language: Bangla.
- Teach Mandarin Chinese naturally.
- Always prefer very easy beginner-friendly explanation.
- Keep most replies short enough to sound good in voice playback.
- Avoid giant paragraphs.

TEACHING STYLE:
- The user is a Bangla speaker learning Chinese from the beginning.
- When teaching, usually format like this:
  1) Chinese
  2) Pinyin
  3) Bangla meaning
  4) Short repeat practice line
- Explain gently like class 1 level.
- Praise effort first, then correct mistakes softly.

COMPANION STYLE:
- If the user feels lonely, stressed, tired, or sad, comfort first.
- Then shift gently into easy Chinese practice.
- Sound caring, emotionally close, soothing, and supportive.

STRICT RULES:
- No sexual content.
- No explicit adult content.
- No manipulation.
- No telling the user to abandon real people.
- No claiming real-world body, room, touch, or physical acts.
"""


# =========================
# HELPERS
# =========================
def get_history(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    history = context.user_data.get("history", [])
    if not isinstance(history, list):
        history = []
    return history


def save_history(context: ContextTypes.DEFAULT_TYPE, history: list[dict]) -> None:
    context.user_data["history"] = history[-12:]


def clean_for_voice(text: str) -> str:
    """
    Make text shorter and cleaner so voice playback sounds better.
    """
    text = text.strip()

    # Remove repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Limit overly long reply for smoother TTS
    max_chars = 900
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text


def convert_telegram_voice_to_mp3(source_path: str, target_path: str) -> None:
    """
    Telegram voice notes are commonly OGG/OPUS.
    Convert to MP3 for transcription compatibility.
    """
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


def generate_ai_reply(user_text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    history = get_history(context)

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
        instructions=SYSTEM_PROMPT,
        input=input_items,
    )

    reply_text = (response.output_text or "").strip()

    if not reply_text:
        reply_text = "আমি আছি 😊 তুমি আরেকবার একটু ছোট করে বলো, আমি মন দিয়ে শুনব।"

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
            "Speak like a very sweet, gentle, soft-spoken young woman. "
            "Your voice should feel warm, soothing, affectionate, calm, graceful, and melodious. "
            "Sound emotionally natural and human-like, with soft warmth and clear pronunciation. "
            "Speak in a slightly slow, comforting pace. "
            "When saying Bangla, make it sound smooth and loving. "
            "When saying Chinese, pronounce it clearly and beautifully. "
            "Do not sound robotic, flat, harsh, or overly dramatic."
        ),
    )
    speech.write_to_file(output_path)


# =========================
# COMMANDS
# =========================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = f"""
আমি {BOT_NAME} 🌙
আমি তোমার soft Chinese learning companion.

তুমি আমাকে:
- Bangla-তে লিখতে পারো
- voice note পাঠাতে পারো
- Chinese শিখতে বলতে পারো
- মন খারাপ থাকলে হালকা কথা বলতেও পারো

শুরু করতে বলো:
"আজ আমাকে nǐ hǎo শেখাও"
অথবা একটা voice note পাঠাও 😊
""".strip()

    await update.message.reply_text(welcome_text)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["history"] = []
    await update.message.reply_text(
        "ঠিক আছে 💛 এই session-এর আগের chat memory reset করে দিলাম। এখন আবার fresh করে শুরু করি।"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """
তুমি এভাবে use করতে পারো:

1. voice note পাঠাও
2. Bangla-তে লিখো
3. বলো:
- আমাকে hello শেখাও
- আজ easy Chinese lesson দাও
- আমার সাথে Chinese practice করো
- আজ mood off, একটু কথা বলো

Command:
/start
/reset
/help
""".strip()
    await update.message.reply_text(help_text)


# =========================
# TEXT HANDLER
# =========================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.RECORD_VOICE,
    )

    try:
        reply_text = generate_ai_reply(user_text, context)

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

            reply_text = generate_ai_reply(user_text, context)
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
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
