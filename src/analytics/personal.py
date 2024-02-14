from dataclasses import dataclass
from src.database import get_categories, select_raw

@dataclass
class PersonalStatistics:
    by_category: dict[str, 'PersonalCategoryStatistics']


@dataclass
class PersonalCategoryStatistics:
    submissions_number: int
    days_with_submissions: int


def _get_category_statistics(submissions, category_id):
    category_submissions = [
        count for _, cat_id, count in submissions if cat_id == category_id
    ]
    return PersonalCategoryStatistics(
        submissions_number=sum(category_submissions),
        days_with_submissions=len(category_submissions),
    )


async def get_personal_statistics(user_id: int) -> PersonalStatistics:
    submissions = await select_raw("""
        SELECT DATE(s.created_at), g.category_id, count(*)
            FROM submissions s
            JOIN goals g ON s.goal_id = g.goal_id
            WHERE s.created_at > now() - interval '1 week' AND s.user_id = :user_id
            GROUP BY DATE(s.created_at), g.category_id
            ORDER BY count(*) DESC
    """, user_id=user_id)

    all_categories = await get_categories()
    all_categories = {
        category.category_id: category for category in all_categories
    }
    user_categories = list(set([category_id for _, category_id, _ in submissions]))

    result = {}

    for category_id in user_categories:
        category_name = all_categories[category_id].name
        result[category_name] = _get_category_statistics(submissions, category_id)

    return PersonalStatistics(by_category=result)
