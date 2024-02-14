from collections import defaultdict
from src.database import get_categories, select_raw


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
