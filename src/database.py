import datetime
import logging
from typing import Optional

from sqlalchemy import (
    func,
    text,
    insert,
    select,
    update,
    delete,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
)
from sqlalchemy.orm import (
    joinedload,
    declarative_base,
)
from async_lru import alru_cache

from src.models import (
    User,
    Goal,
    Event,
    Category,
    Submission,
    Leaderboard,
    ExternalPlatform,
    ExternalPlatformConnection,
)


# Setting up logger
logger = logging.getLogger(__name__)

DB_ENGINE = None


async def init_db(database_url: str):
    global DB_ENGINE

    try:
        DB_ENGINE = create_async_engine(
            database_url,
            echo=False,
        )
        
        # create tables
        # async with DB_ENGINE.begin() as conn:
        #     from src.models import Base
        #     await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize database")


async def close_db():
    if DB_ENGINE is not None:
        await DB_ENGINE.dispose()


async def save_user_personal_details(discord_user, email, name) -> User:
    user = await ensure_user(discord_user)
    async with DB_ENGINE.begin() as conn:
        cursor = await conn.execute(update(User).where(
            User.user_id==user.user_id).values(
                email=email, name=name
        ).returning(User))

        user = cursor.fetchone()
        
        get_user.cache_invalidate(user.user_id)
        
        logger.info(f"Updated user name and email for : {user.username} with ID {user.user_id}")
        return user


async def _new_user(user_id, username) -> User:
    """ Create new user
    Always use ensure_user instead of this function
    It's not any faster to use this function,
      because if the user exists, we will not create a new user.
    """
    async with DB_ENGINE.begin() as conn:
    
        cursor = await conn.execute(insert(User).values(
            user_id=user_id,
            username=username,
        ).returning(User))

        user = cursor.fetchone()
        
        get_user.cache_invalidate(user_id)
        
        logger.info(f"New user created: {username} with ID {user_id}")
        return user


async def new_submission(
    user_id, goal_id, proof_url, amount, 
    created_at=None, voice_channel: str = None,
) -> Submission:
    async with DB_ENGINE.begin() as conn:
        cursor = await conn.execute(insert(Submission).values(
            user_id=user_id,
            goal_id=goal_id,
            proof_url=proof_url,
            created_at=created_at or datetime.datetime.now(datetime.UTC),
            amount=amount,
            is_voice=voice_channel is not None,
            voice_channel=voice_channel,
        ).returning(Submission))

        submission = cursor.fetchone()
        logger.info(f"New submission for user {user_id}")
        return submission


async def new_goal(user_id, category_id,goal_description, metric, target, frequency) -> Goal:
    async with DB_ENGINE.begin() as conn:
        cursor = await conn.execute(insert(Goal).values(
            user_id=user_id,
            category_id=category_id,
            goal_description=goal_description,
            metric=metric,
            target=target,
            frequency=frequency,
        ).returning(Goal))

        goal = cursor.fetchone()
        
        get_goal.cache_invalidate(category_id, user_id)
        
        logger.info(f"Attempt to set New Goal for {user_id}")
        return goal


@alru_cache(maxsize=1000) 
async def get_category(text_channel) -> Optional[Category]:
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(Category).where(Category.text_channel == text_channel))).first()


@alru_cache(maxsize=1000)
async def get_category_by_name(name) -> Optional[Category]:
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(Category).where(
            Category.name == name))).first()


@alru_cache(maxsize=1000)
async def get_category_for_voice(voice_channel) -> Optional[Category]:
    # 30_days_ml_5 -> 30_days_ml
    voice_channel_parts = voice_channel.rsplit('_', 1)
    if voice_channel_parts[-1].isdigit():
        voice_channel = voice_channel_parts[0]
        
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(Category).where(
            Category.voice_channel == voice_channel))).first()


@alru_cache(maxsize=1000)
async def get_user(user_id) -> Optional[User]:
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(User).where(User.user_id == user_id))).first()


@alru_cache(maxsize=1000)
async def get_goal(category_id, user_id) -> Optional[Goal]:
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(
            select(Goal)
                .where(Goal.category_id == category_id)
                .where(Goal.user_id == user_id)
                .order_by(Goal.created_at.desc())
        )).first()
    

async def get_user_goals(user_id):
    async with DB_ENGINE.begin() as conn:
        all_goals = (await conn.execute(
            select(Goal)
                .where(Goal.user_id == user_id)
                .where(Goal.active == True)
                .order_by(Goal.created_at)
        )).fetchall()

        goals_by_category = {
            goal.category_id: goal
            for goal in all_goals
        }

        return list(goals_by_category.values())


async def ensure_user(discord_user) -> User:
    user = await get_user(discord_user.id)
    
    if user is None:
        user = await _new_user(
            user_id=discord_user.id,
            username=discord_user.name,
        )
    
    return user
    


async def get_submission_leaderboard():
    return []


async def select_raw(query, **params):
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(text(query), params)).fetchall()


@alru_cache()
async def get_categories():
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(Category))).fetchall()


async def get_submissions_without_proof(user_id, window_hours=24) -> list[Submission]:
    """ Returns submissions for the last 24 hours without proof. """
    min_created_at = datetime.datetime.now() - datetime.timedelta(
        hours=window_hours)
    
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(
            select(Submission)
                .where(Submission.user_id == user_id)
                .where(Submission.created_at > min_created_at)
                .where(Submission.proof_url == None)
                .options(joinedload(Submission.goal))
        )).fetchall()


async def update_submission_proof(submission_id, proof_url):
    async with DB_ENGINE.begin() as conn:
        await conn.execute(update(Submission).where(
            Submission.submission_id==submission_id).values(
                proof_url=proof_url,
        ))

    logger.info(f"Updated proof for submission {submission_id}")


async def update_user_last_llm_submission(user_id, last_llm_submission):
    async with DB_ENGINE.begin() as conn:
        await conn.execute(update(User).where(
            User.user_id==user_id).values(
                last_llm_submission=last_llm_submission,
        ))

    logger.info(f"Updated last_llm_submission for user {user_id}")


async def update_user_timezone_shift(user_id, timezone_shift):
    async with DB_ENGINE.begin() as conn:
        await conn.execute(update(User).where(
            User.user_id==user_id).values(
                time_zone_shift=timezone_shift,
        ))

    logger.info(f"Updated time_zone_shift for user {user_id} (shift={timezone_shift})")


async def get_users_without_timezone_shift():
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(
            select(User.user_id, User.time_zone_shift)
                .where(User.time_zone_shift == None)
        )).fetchall()


async def get_users_to_notify(timezone_shift: int):
    logger.info(f"Fetching users to notify for timezone {timezone_shift}")

    stmt = (
        select(User.username, func.count(Submission.submission_id))
        .join(Goal, User.user_id == Goal.user_id)
        .outerjoin(Submission, (User.user_id == Submission.user_id) & (Submission.created_at > text("NOW() - INTERVAL '1 DAY'")))
        .where(User.time_zone_shift > timezone_shift - 2)
        .where(User.time_zone_shift < timezone_shift + 2)
        .group_by(User.user_id)
        .having(func.count(Submission.submission_id) == 0)
    )
    
    async with DB_ENGINE.begin() as conn:
        rows = (await conn.execute(stmt)).fetchall()

    return rows


async def list_leaderboards() -> list[Leaderboard]:
    async with DB_ENGINE.begin() as conn:
        leaderboards = await conn.execute(select(Leaderboard))
        return leaderboards.fetchall()


async def update_leaderboard_last_sent(leaderboard_id, timestamp):
    async with DB_ENGINE.begin() as conn:
        await conn.execute(update(Leaderboard).where(
            Leaderboard.leaderboard_id==leaderboard_id).values(
                last_sent=timestamp,
        ))

    logger.info(f"Updated last_sent for leaderboard {leaderboard_id}")


async def list_submissions_by_voice_channel(voice_channel) -> Submission:
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(
            select(Submission)
                .where(Submission.voice_channel == voice_channel)
                .order_by(Submission.created_at)
        )).fetchall()
    
    
async def create_event(user_id, event_type, payload):
    async with DB_ENGINE.begin() as conn:
        cursor = await conn.execute(insert(Event).values(
            user_id=user_id,
            event_type=event_type,
            payload=payload,
        ).returning(Event))

        event = cursor.fetchone()
        logger.info(f"New event for user {user_id}: {event_type}")
        return event


@alru_cache(maxsize=1000)
async def get_external_platform(platform_name) -> Optional[ExternalPlatform]:
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(ExternalPlatform).where(
            ExternalPlatform.platform_name == platform_name))).first()
    

@alru_cache(maxsize=1000)
async def get_external_platform_by_id(platform_id) -> Optional[ExternalPlatform]:
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(ExternalPlatform).where(
            ExternalPlatform.platform_id == platform_id))).first()


async def create_external_platform_connection(user_id, platform_id, user_name, user_data=None):
    async with DB_ENGINE.begin() as conn:
        cursor = await conn.execute(insert(ExternalPlatformConnection).values(
            user_id=user_id,
            platform_id=platform_id,
            user_name=user_name,
            user_data=user_data,
        ).returning(ExternalPlatformConnection))

        connection = cursor.fetchone()
        logger.info(f"New external platform connection for user {user_id}")
        return connection


async def list_external_platform_connections():
    async with DB_ENGINE.begin() as conn:
        return (await conn.execute(select(ExternalPlatformConnection))).fetchall()
    

async def set_external_platform_connection_user_data(connection_id, user_data):
    async with DB_ENGINE.begin() as conn:
        await conn.execute(update(ExternalPlatformConnection).where(
            ExternalPlatformConnection.connection_id == connection_id,
        ).values(
            user_data=user_data,
        ))

    logger.info(f"Updated user data for external platform connection {connection_id}")