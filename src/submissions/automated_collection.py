import asyncio
import datetime
import logging
import os
from typing import Optional
import uuid

import aiohttp
import discord
from dotenv import load_dotenv

from src.models import ExternalPlatformConnection
from src.database import (
    get_goal,
    get_user,
    new_submission,
    get_category_by_name,
    get_external_platform_by_id,
    list_external_platform_connections,
    set_external_platform_connection_user_data,
)


load_dotenv()

logger = logging.getLogger(__name__)

SUBMISSION_CHANNEL_ID = os.environ['SUBMISSION_CHANNEL_ID']


# Unauthenticated session
LEETCODE_COOKIES = {
    'csrftoken': 'htoYfswx1ql6Vvo2UE7jtqgxq1PcA7UMJRAFlH9R6RT1LjCtj3ZKVFKj5ktr5e91',
    '__cf_bm': 'pA2bZoRg09DtM32TvlPDtmbVTNN6s9g0BG2oNCMj1k8-1719773173-1.0.1.1-X_i0D3QzLexa0BmTgxepyISc4TX28a2RQNBngO_wZMmIuzjWDhmMceCbZVuZIeC.Je89uMPf8OSivdGYauvjUg',
    '_gid': 'GA1.2.1547655166.1719773177',
    '_gat': '1',
    'gr_user_id': '9dad0a8a-7a1c-4967-8dc1-f2a4bee9014b',
    '87b5a3c3f1a55520_gr_session_id': '908a1482-2a6e-4283-a6e8-6c77afc524d6',
    '87b5a3c3f1a55520_gr_session_id_sent_vst': '908a1482-2a6e-4283-a6e8-6c77afc524d6',
    'INGRESSCOOKIE': 'ed27c6133b195b82d57a97928c2c76fd|8e0876c7c1464cc0ac96bc2edceabd27',
    '__stripe_mid': 'c7318bc6-3ccf-4158-bf25-9e70cc1992cc1fec19',
    '__stripe_sid': 'b22d3c22-0804-47e4-ab12-9cad6c1af17c0ca362',
    '_ga': 'GA1.1.1004498456.1719773176',
    '_dd_s': 'rum=0&expire=1719774090724',
    '_ga_CDRWKZTDEX': 'GS1.1.1719773175.1.1.1719773218.17.0.0',
}

LEETCODE_HEADERS = {
    'accept': '*/*',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'authorization': '',
    'baggage': 'sentry-environment=production,sentry-release=e22a6fe5,sentry-transaction=%2Fu%2F%5Busername%5D,sentry-public_key=2a051f9838e2450fbdd5a77eb62cc83c,sentry-trace_id=465f5bb1ee31416c8e566e106a053c6e,sentry-sample_rate=0.03',
    'content-type': 'application/json',
    'dnt': '1',
    'origin': 'https://leetcode.com',
    'priority': 'u=1, i',
    'random-uuid': 'faa5f9dd-9c77-8f73-dd04-8675e905301d',
    'referer': 'https://leetcode.com/u/neal_wu/',
    'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'sentry-trace': '465f5bb1ee31416c8e566e106a053c6e-aa0cd949fd93d67a-0',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'x-csrftoken': 'htoYfswx1ql6Vvo2UE7jtqgxq1PcA7UMJRAFlH9R6RT1LjCtj3ZKVFKj5ktr5e91',
}


async def _fetch_external_platform_data(
    session: aiohttp.ClientSession, 
    platform_connection: ExternalPlatformConnection
) -> Optional[dict[str, int]]:
    json_data = {
        'query': '\n    query userProfileUserQuestionProgressV2($userSlug: String!) {\n  userProfileUserQuestionProgressV2(userSlug: $userSlug) {\n    numAcceptedQuestions {\n      count\n      difficulty\n    }\n    numFailedQuestions {\n      count\n      difficulty\n    }\n    numUntouchedQuestions {\n      count\n      difficulty\n    }\n    userSessionBeatsPercentage {\n      difficulty\n      percentage\n    }\n  }\n}\n    ',
        'variables': {
            'userSlug': platform_connection.user_name,
        },
        'operationName': 'userProfileUserQuestionProgressV2',
    }
    
    try:
        result = await session.post("/graphql/", json=json_data)
    except aiohttp.ClientError:
        logger.exception("Failed to fetch data from external platform")
        return None
    
    if result.status != 200:
        logger.error("Got unexpected status code %s", result.status)
        return None
    
    data = await result.json()

    accepted = data['data']['userProfileUserQuestionProgressV2']['numAcceptedQuestions']
    counts = {
        row['difficulty']: int(row['count']) for row in accepted
    }

    return counts



async def _collect_submissions_automatically(client: discord.Client):
    logger.info("Fetching data from external platforms")
    session = aiohttp.ClientSession(
        base_url="https://leetcode.com/",
        headers=LEETCODE_HEADERS,
        cookies=LEETCODE_COOKIES
    )

    notification_channel = await client.fetch_channel(SUBMISSION_CHANNEL_ID)

    # TODO: link platforms to categories in database
    categories = {
        1: await get_category_by_name('_automated_LeetCode'),
    }
    
    external_connection = await list_external_platform_connections()
    
    async with session:            
        for external_connection in external_connection:
            data = await _fetch_external_platform_data(
                session, external_connection
            )
            if data is None:
                continue

            if external_connection.user_data:
                category = categories[external_connection.platform_id]
                diffs_sum = sum(data.values()) - sum(external_connection.user_data.values())

                if diffs_sum == 0:
                    logger.info(f"No changes in platform {external_connection.platform_id} for user {external_connection.user_id}")
                    continue

                goal = await get_goal(category.category_id, external_connection.user_id)
                if not goal:
                    logger.error(f"Goal not found for user {external_connection.user_id} and category {category.category_id}")
                    continue

                await new_submission(
                    user_id=external_connection.user_id,
                    goal_id=goal.goal_id,
                    proof_url=None,
                    amount=diffs_sum,
                )

                platform = await get_external_platform_by_id(external_connection.platform_id)
                if not platform:
                    logger.error(f"External platform {external_connection.platform_id} not found")
                    continue

                user = await get_user(external_connection.user_id)

                problems_form = "prolem" if diffs_sum == 1 else "problems"
                await notification_channel.send(f"User {user.username} solved {diffs_sum} {problems_form} on {platform.platform_name}")

            await set_external_platform_connection_user_data(
                connection_id=external_connection.connection_id,
                user_data=data,
            )


async def collect_submissions_automatically(client: discord.Client):
    await client.wait_until_ready()

    while True:
        await _collect_submissions_automatically(client)

        await asyncio.sleep(60 * 60 * 24)
