import pytest
import asyncio
from src.discord_app import init, client 

@pytest.mark.asyncio
async def test_bot_starts():
    # Set up a listener for the 'on_ready' event to signal bot readiness
    ready_signal = asyncio.Event()

    @client.event
    async def on_ready():
        ready_signal.set()  # Signal that the bot is ready

    # Start the bot in the background.
    bot_task = asyncio.create_task(init())

    # Wait for the ready signal with a timeout
    await asyncio.wait_for(ready_signal.wait(), timeout=10)

    # Check if the bot is connected and logged in
    assert ready_signal.is_set()

    # Close the bot connection
    await client.close()
    await bot_task