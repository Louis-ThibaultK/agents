import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, qwen, silero, openai

load_dotenv()
logger = logging.getLogger("voice-assistant")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    dg_model = "nova-2-general"
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        # use a model optimized for telephony
        dg_model = "nova-2-phonecall"

    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
         ## openai
        stt=openai.STT(base_url="https://api.chatanywhere.tech/v1",language="zh",model="whisper-1"),
        ## qwen2
        ## stt=openai.STT(base_url="http://10.218.127.99:3000/", language="zn"),
        ## openai
        llm=openai.LLM(base_url="https://api.chatanywhere.tech/v1"),
        ## qwen2
        ## llm=openai.LLM(base_url="http://10.218.127.29:11434/v1/", model="qwen2:72b"),

        tts=qwen.TTS(
            seed=42,
            style_type="中文女",
            base_url="http://10.218.127.100:3000/instruct/synthesize",
            prompt="A girl speaker with a brisk pitch, an enthusiastic speaking pace, and a upbeat emotional demeanor.",
        ),
        chat_ctx=initial_ctx,
    )

    assistant.start(ctx.room, participant)

    # listen to incoming chat messages, only required if you'd like the agent to
    # answer incoming messages from Chat
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        chat_ctx = assistant.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        stream = assistant.llm.chat(chat_ctx=chat_ctx)
        await assistant.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    await assistant.say("你好", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
