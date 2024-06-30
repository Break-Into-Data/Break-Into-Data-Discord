import os
import datetime
from collections import defaultdict

import dotenv

from src.database import get_categories, list_leaderboards, list_submissions_by_voice_channel, select_raw, update_leaderboard_last_sent
from src.models import Leaderboard


dotenv.load_dotenv()


LEADERBOARD_CHANNEL_ID = os.environ['DISCORD_SETTINGS_CHANNEL_ID']


async def _get_category_leaderboard(category_id: int):
    submissions = await select_raw("""
        SELECT s.user_id, DATE(s.created_at), count(*)
            FROM submissions s
            JOIN goals g ON s.goal_id = g.goal_id
            WHERE s.created_at > now() - interval '1 week' AND g.category_id = :category_id
            GROUP BY s.user_id, DATE(s.created_at)
            ORDER BY count(*) DESC
    """, category_id=category_id)

    sumbissions_by_user = defaultdict(int)
    days_by_user = defaultdict(int)

    for user_id, date, count in submissions:
        sumbissions_by_user[user_id] += count
        days_by_user[user_id] += 1

    return sorted([
        (user_id, sumbissions_by_user[user_id], days_by_user[user_id])
        for user_id in sumbissions_by_user
    ], key=lambda x: x[2], reverse=True)


async def get_weekly_leaderboard():
    categories = await get_categories()

    leaderboards = {}

    for category in categories:
        leaderboard = await _get_category_leaderboard(category.category_id)
        leaderboards[category.name] = leaderboard

    #print(leaderboards)

    return leaderboards


def _should_send_leaderboard(leaderboard, now):
    return (
        leaderboard.last_sent is not None and 
        now - leaderboard.last_sent < datetime.timedelta(hours=24)
    )


async def _get_leaderboard_message(leaderboard: Leaderboard, discord_client):
    current_date = datetime.now().strftime("%d-%b-%Y")
    
    voice_channels = leaderboard.voice_channels.split(',')
    
    hours = []
    
    for voice_channel in voice_channels:
        channel_submissions = list_submissions_by_voice_channel(
            voice_channel)
        
        hours.extend(channel_submissions)
    
    hours_by_user = defaultdict(float)
    days_by_user = defaultdict(set)
    for submission in hours:
        hours_by_user[submission.user_id] += submission.amount / 60
        days_by_user[submission.user_id].add(submission.created_at.date())
        
    board_data = sorted([
        (user_id, hours_by_user[user_id], len(days_by_user[user_id]))
        for user_id in hours_by_user
    ], key=lambda x: x[1], reverse=True)

    msg_parts = [f">>> **Weekly leaderboard {leaderboard.name}: {current_date}**\n"]
    for user_id, hours, days in board_data:
        username = (await discord_client.fetch_user(user_id)).global_name
        msg_parts.append(f"{username}: {hours:.2f} hours, {days} active days")
    msg_parts.append("")

    msg = "\n".join(msg_parts)
    
    return msg


async def send_voice_leaderboards(discord_client):
    now = datetime.datetime.now(datetime.timezone.utc)
    
    leaderboards = await list_leaderboards()
    
    leaderboard_channel = await discord_client.fetch_channel(LEADERBOARD_CHANNEL_ID)
    
    for leaderboard in leaderboards:
        if not _should_send_leaderboard(leaderboard, now):
            continue
        
        leaderboard_message = _get_leaderboard_message(leaderboard)
        
        await leaderboard_channel.send(leaderboard_message)
        await update_leaderboard_last_sent(leaderboard.leaderboard_id, now)
