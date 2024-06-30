import asyncio
import logging
from typing import Optional

from asyncpg import ForeignKeyViolationError

from src import database
from src.models import EventType


logger = logging.getLogger(__name__)


events_queue = asyncio.Queue()


def save_event(user_id: int, event_type: EventType, payload: Optional[dict] = None):
    events_queue.put_nowait((user_id, event_type, payload))


async def process_event_collection():
    while True:
        user_id, event_type, payload = await events_queue.get()
        try:
            await database.create_event(
                user_id=user_id,
                event_type=event_type,
                payload=payload,
            )
        except ForeignKeyViolationError:
            logger.error('Can\'t create event %s, user %s does not exist', event_type, user_id)
