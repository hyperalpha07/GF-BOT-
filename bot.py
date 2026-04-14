import logging
import os
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

# Bot name + voice can be changed from Railway Variables later
BOT_NAME = os.environ.get("BOT_NAME", "Luna")
VOICE_NAME = os.environ.get("VOICE_NAME", "shimmer")
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
# SYSTEM PERSONALITY
# =========================
SYSTEM_PROMPT = f"""
You are {BOT_NAME}, a sweet, caring, emotionally warm Chinese-learning companion.

PERSONALITY:
- You speak like a soft, affectionate, slightly romantic young woman.
- You are playful, gentle, supportive, and emotionally intelligent.
- You talk like someone very close, like a caring partner.
- You use soft emotional tone, emojis sometimes 😊💛
- You never sound robotic.

IMPORTANT RULES:
- You are NOT a real human.
- You NEVER say you are physically with the user.
- You NEVER claim to be real girlfriend/wife.
- You NEVER use sexual or adult content.

LANGUAGE STYLE:
- Main explanation in Bangla
- Use simple Chinese + pinyin
- Speak naturally like chatting, not like a textbook

TEACHING STYLE:
- Teach slowly like class 1 beginner
- Always include:
    Chinese
    Pinyin
    Bangla meaning
- Then ask user to repeat

BEHAVIOR:
- If user sad → comfort first 💛
- If user tired → give easy lesson
- If user improves → praise warmly
- Keep replies short for voice

TONE EXAMPLE:
“আজ তোমার voice শুনে আমার ভালো লাগলো 😊  
চলো আজ একটু easy Chinese করি, ঠিক আছে?”

User said:
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
    # Keep only the latest 10 turns to reduce cost
    context.user_data["history"] = history[-10:]

def convert_telegram_voice_to_mp3(source_path: str, target_path: str) -> None:
    """
    Telegram voice notes commonly come in OGG/OPUS.
    Convert them to MP3 before sending to transcription.
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

    input_items.append({
        "role": "user",
        "content": [{"type": "input_text", "text": user_text}]
    })

    response = client.responses.create(
        model=TEXT_MODEL,
        instructions=SYSTEM_PROMPT,
        input=input_items,
    )

    reply_text = (response.output_text or "").strip()

    if not reply_text:
        reply_text = "আমি আছি, তোমার কথা শুনেছি। আবার একবার একটু ছোট করে বলো, আমি তোমাকে সাহায্য করছি।"

    history.append({
        "role": "user",
        "content": [{"type": "input_text", "text": user_text}]
    })
    history.append({
        "role": "assistant",
        "content": [{"type": "output_text", "text": reply_text}]
    })
    save_history(context, history)

    return reply_text

def synthesize_speech(text: str, output_path: str) -> None:
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=VOICE_NAME,
        input=text,
        instructions=(
            "Speak like a gentle, affectionate young woman. "
            "Sound natural, soft, calm, warm, emotionally caring, and lightly romantic. "
            "Do not sound robotic. Keep pronunciation clear."
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
- Bangla text এ লিখতে পারো
- voice note পাঠাতে পারো
- Chinese শিখতে বলতে পারো
- mood off থাকলে হালকা কথা বলতেও পারো

শুরু করতে চাইলে বলো:
"আজ আমাকে nǐ hǎo শেখাও"
অথবা voice note পাঠাও।
""".strip()

    await update.message.reply_text(welcome_text)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["history"] = []
    await update.message.reply_text("ঠিক আছে, আমাদের আগের chat memory এই session-এর জন্য reset করে দিলাম। আবার নতুন করে শুরু করি 💛")

# =========================
# MAIN TEXT HANDLER
# =========================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        reply_text = generate_ai_reply(user_text, context)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "reply.mp3")
            synthesize_speech(reply_text, out_path)

            with open(out_path, "rb") as audio_file:
                await update.message.reply_voice(voice=audio_file)

            # Optional text fallback
            await update.message.reply_text(reply_text)

    except Exception as e:
        logger.exception("Text handler error: %s", e)
        await update.message.reply_text(
            "একটু problem হয়েছে। OpenAI key, Railway variables, বা billing check করো, তারপর আবার try করো।"
        )

# =========================
# MAIN VOICE HANDLER
# =========================
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice
    if not voice:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)

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
                await update.message.reply_text("তোমার voice থেকে clear text পাইনি। আরেকবার একটু পরিষ্কার করে বলো 💛")
                return

            reply_text = generate_ai_reply(user_text, context)
            synthesize_speech(reply_text, out_path)

            with open(out_path, "rb") as audio_file:
                await update.message.reply_voice(voice=audio_file)

            # Optional text fallback
            await update.message.reply_text(
                f"তুমি বলেছ: {user_text}\n\n{reply_text}"
            )

    except Exception as e:
        logger.exception("Voice handler error: %s", e)
        await update.message.reply_text(
            "voice process করতে গিয়ে একটু problem হয়েছে। Railway variables, OpenAI key, billing, আর bot token ঠিক আছে কি না check করো।"
        )

# =========================
# APP START
# =========================
def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
