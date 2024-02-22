import os
import sys
import asyncio
from datetime import (
    datetime, 
    timedelta,
)

from src.discord_app import process_discord_message
from src.database import init_db

import dotenv
import discord

dotenv.load_dotenv()

DISCORD_TOKEN = os.environ['DISCORD_BOT_TOKEN']
DISCORD_SERVER_ID = os.environ['DISCORD_SERVER_ID']
SUBMISSION_CHANNEL_ID = sys.argv[1]  # ['1197314741534736464', '1197259793266638970', '1168693561429069834']
BATCH_LIMIT = 100
FETCH_DAYS = 7

client = discord.Client(
    intents=discord.Intents.all(),
)


async def iterate_channel_messages(channel: discord.TextChannel, batch_limit: int = 100, days: int = 7):
    after = datetime.utcnow() - timedelta(days=days)

    while True:
        async for message in channel.history(limit=batch_limit, after=after, oldest_first=True): 
            yield message
            after = message



async def backfill_submissions():  
    channel = await client.fetch_channel(SUBMISSION_CHANNEL_ID)

    async for message in iterate_channel_messages(channel, batch_limit=BATCH_LIMIT, days=FETCH_DAYS):
        await process_discord_message(message, is_backfill=True)
                

@client.event
async def on_ready():
    await backfill_submissions()


async def init():
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise Exception("Please set the DATABASE_URL environment variable")

    await init_db(DATABASE_URL)

    discord.utils.setup_logging()
    return await client.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(init())
