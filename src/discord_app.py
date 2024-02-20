import asyncio
import os
import logging
from dataclasses import dataclass
from datetime import (
    datetime, 
    timedelta, 
    timezone
    )
import dotenv
import discord
from discord import app_commands
from discord.ext import tasks

from src.database import (
    get_category_by_name,
    get_category_for_voice,
    get_user_goals,
    init_db,
    get_goal,
    new_goal,
    ensure_user,
    get_category,
    new_submission,
)
from src.analytics.leaderboard import get_weekly_leaderboard
from src.analytics.personal import get_personal_statistics
from src.submissions import process_submission_message
from src.onboarding import OnboardingView


dotenv.load_dotenv()

DISCORD_TOKEN = os.environ['DISCORD_BOT_TOKEN']
SETTINGS_CHANNEL_ID = os.environ['DISCORD_SETTINGS_CHANNEL_ID']
GENERAL_CHANNEL_ID = os.environ['DISCORD_GENERAL_CHANNEL_ID']
DISCORD_SERVER_ID = os.environ['DISCORD_SERVER_ID']
SUBMISSION_CHANNEL_ID = os.environ['SUBMISSION_CHANNEL_ID']

client = discord.Client(
    intents=discord.Intents.all(),
)

logging.basicConfig(
    filename='bot.log',
    encoding='utf-8', 
    level=logging.DEBUG, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(filename)s:line %(lineno)d: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Track:
    name: str

    questions_needed: list
    default_tracking_metric: str
    default_daily_target: str


TRACKS = {
    track.name: track for track in [
        Track(name="Fitness",
              questions_needed=['description', 'metric', 'target', 'frequency'],
              default_tracking_metric="e.g. miles, minutes ",
              default_daily_target='e.g.30'),
        Track(name="Coding",
              questions_needed=['description', 'metric', 'target', 'frequency'],
              default_tracking_metric="e.g. problems, minutes", 
              default_daily_target='e.g. 3 or 5'),
        Track(name="Studying",
              questions_needed=['description', 'metric', 'target', 'frequency'],
              default_tracking_metric="e.g. minutes, chapters", 
              default_daily_target='60 or 2'),
        Track(name="Meditation",
              questions_needed=['description', 'metric', 'target', 'frequency'],
              default_tracking_metric="e.g. minutes, sessions",
              default_daily_target='60 or 2'),
        Track(name="Content Creation",
              questions_needed=['description', 'metric', 'target', 'frequency'],
              default_tracking_metric="e.g. posts, videos, blogs", 
              default_daily_target='1 or 2'),
        Track(name="Other",
              questions_needed=['description', 'metric', 'target', 'frequency'],
              default_tracking_metric="Smiles", 
              default_daily_target=10),
    ]
}


class TrackSettingsView(discord.ui.View):
    """ Creates a view with the buttons for each track """
    def __init__(self):
        super().__init__(timeout=None)

        for track in TRACKS.values():
            self.add_button(track)

    def add_button(self, track: Track):
        btn = discord.ui.Button(label=track.name,
                                style=discord.ButtonStyle.secondary, custom_id=track.name)
        self.add_item(btn)

    async def interaction_check(self, interaction: discord.Interaction):
        if not interaction.type == discord.InteractionType.component:
            return False
        track = TRACKS.get(interaction.data['custom_id'])
        if not track:
            interaction.response.send_message(
                "Something went wrong, please try again", ephemeral=True)
            return False

        modal = TrackSettingsModal(track)
        await interaction.response.send_modal(modal)

        return True


class TrackSettingsModal(discord.ui.Modal):
    """ Adds user's inputs in the goals table """
    def __init__(self, track: Track):
        super().__init__(title=track.name)
        self.track = track
        #print(track.questions_needed)

        if 'description' in track.questions_needed:
            # New field for their
            self.description_input = discord.ui.TextInput(
                label="Description",
                required=True,
                placeholder="Describe your goal!")
            self.add_item(self.description_input)


        if 'metric' in track.questions_needed:
            self.metric_input = discord.ui.TextInput(
                label="What is your metric?",
                placeholder=track.default_tracking_metric)
            self.add_item(self.metric_input)


        target_map = {
            'Coding': 'Daily coding goals? 5 problems/2 hours',
            'Meditation': 'Set your meditation goal',
            'Fitness': 'What is your exercise target?',
            'Studying': 'Set your study goal',
            'Content Creation': 'How much content to make?',
            'Other': 'Set your Number to measure.'
        }

        if 'target' in track.questions_needed:
            self.daily_target_input = discord.ui.TextInput(
                label=target_map[track.name],
                placeholder=f"{track.default_daily_target} (it must be a number)")
            self.add_item(self.daily_target_input)

        # New field for frequency
        if 'frequency' in track.questions_needed:
            self.frequency_input = discord.ui.TextInput(
                label="What is the frequency per week? ",
                placeholder="e.g. 4, 5 (times a week) ")
            self.add_item(self.frequency_input)


    async def on_submit(self, interaction: discord.Interaction):
        """ Makes sure the user input a number and not a string """
        logging.info(f'Goal submission attempt by {interaction.user}')
        try:
            daily_target = int(self.daily_target_input.value.strip())
        except ValueError:
            await interaction.response.send_message(
                "Daily target must be a valid number, please try again",
                ephemeral=True)
            return

        await interaction.response.defer(thinking=True, ephemeral=True)

        user = await ensure_user(interaction.user)

        # TODO: get category by track
        category = await get_category_by_name(self.track.name)


        if category is None:
            await interaction.followup.send(
                "Something went wrong, please try later", ephemeral=True)
            return


        await new_goal(
            user_id=user.user_id,
            category_id=category.category_id,
            goal_description=self.description_input.value if 'description' in self.track.questions_needed else '',
            metric=self.metric_input.value if 'metric' in self.track.questions_needed else self.track.name,
            target=daily_target,
            frequency=self.frequency_input.value if 'frequency' in self.track.questions_needed else 'daily'
        )

        await interaction.followup.send("Your settings were updated!", ephemeral=True)
        logging.info(f'Goal successfully updated for user {interaction.user}')


@client.event
async def on_ready():
    """ Runs when the client becomes ready """
    logging.info(f'Logged in as {client.user}')

    await tree.sync(guild=discord.Object(id=DISCORD_SERVER_ID))
    channel = await client.fetch_channel(SETTINGS_CHANNEL_ID)
    view = TrackSettingsView()
    logging.info(f'Checking for existing message with goal buttons.')
    async for message in channel.history(limit=1):
        if message.author == client.user and 'Pick your goal:' in message.content:
            # Found an existing message to update
            await message.edit(content='Pick your goal:', view=view)
            logging.info(f'Updated existing message with goal buttons.')
            break
        else:
            # No existing message found
            await channel.send('Pick your goal:', view=view)
            logging.info(f'No existing message. Sent new message.')
        
    await send_weekly_leaderboard.start()



async def process_discord_message(message: discord.Message):
    if message.author == client.user:
        return

    if str(message.channel.id) == SUBMISSION_CHANNEL_ID and message.content:
        await process_submission_message(message)
    
    if message.attachments:
        user = await ensure_user(message.author)

        category = await get_category(message.channel.name)
        if category is None:
            return

        goal = await get_goal(category.category_id, user.user_id)
        if goal is None:
            return

        for attchemnt in message.attachments:
            await new_submission(
                user_id=user.user_id,
                goal_id=goal.goal_id,
                proof_url=attchemnt.url,
                amount=0,
            )


@client.event
async def on_message(message):
    await process_discord_message(message)


# map from a user to it's voice channel join time
VOICE_CHANNELS_JOIN_TIME = {}


async def process_voice_channel_activity(member, before, after):
    member_joins_channel = before.channel is None and after.channel is not None
    member_leaves_channel = before.channel is not None and (after.channel is None or after.channel.name != before.channel.name)

    user = await ensure_user(member)

    if member_joins_channel:
        VOICE_CHANNELS_JOIN_TIME[user.user_id] = datetime.utcnow()
        logging.debug(f'Voice channel activity: {member} joined {after.channel.name}')

    if member_leaves_channel:
        logging.debug(f'Voice channel activity: {member} left {before.channel.name}')
        time_joined = VOICE_CHANNELS_JOIN_TIME.pop(user.user_id, None)
        if time_joined is None:
            return
        time_left = datetime.utcnow()
        time_spent = time_left - time_joined

        category = await get_category_for_voice(before.channel.name)
        if category is None:
            return

        goal = await get_goal(category.category_id, user.user_id)
        if goal is None:
            return

        await new_submission(
            user_id=user.user_id,
            goal_id=goal.goal_id,
            proof_url=None,
            amount=time_spent.seconds // 60,
        )


@client.event
async def on_voice_state_update(member, before, after):
    print(member, before, after)
    await process_voice_channel_activity(member, before, after)


@client.event
async def on_error(event, *args, **kwargs):
    logging.error(f'Unhandled error in {event}:', exc_info=True)


tree = app_commands.CommandTree(client)


@tree.command(
    name="submit",
    description="Submit your daily progress",
    guild=discord.Object(id=DISCORD_SERVER_ID),
)
async def submit_command(interaction, amount: int):
    logging.info(f'Processing submit command from {interaction.user}')
    # TODO: handle database errors

    await interaction.response.defer(thinking=True, ephemeral=True)

    user, category = await asyncio.gather(
        ensure_user(interaction.user),
        get_category(interaction.channel.name)
    )

    if category is None:
        await interaction.followup.send(
            "You can't submit in this channel",
            ephemeral=True,
        )
        return

    goal = await get_goal(category.category_id, user.user_id)

    if goal is None:
        await interaction.followup.send(
            f'You don\'t have an active goal in category "{category.name}"',
            ephemeral=True,
        )
        return

    await new_submission(
        user_id=user.user_id,
        goal_id=goal.goal_id,
        proof_url=None,
        amount=amount,
    )

    logging.debug(f'Submit command processed for {interaction.user}')
    await interaction.followup.send("Your progress is saved!", ephemeral=True)


@tree.command(
    name="stats",
    description="Get personal stats for the last week",
    guild=discord.Object(id=DISCORD_SERVER_ID),
)
async def stats_command(interaction):
    await interaction.response.defer(ephemeral=False, thinking=True)

    statistics = await get_personal_statistics(interaction.user.id)

    msg_parts = [
        f"**Personal stats for the last week **\n"
        f"**{interaction.user.global_name}**\n"
    ] + [
        f"**{category_name}**:\n"
        f"Submissions: {category_stats.submissions_number}\n"
        f"Active days: {category_stats.days_with_submissions}\n"
        for category_name, category_stats in statistics.by_category.items()
    ]

    msg = "\n".join(msg_parts)

    await interaction.followup.send(msg, ephemeral=False)


@tree.command(
        name="backfill",
        description="To get existing users name and email",
        guild=discord.Object(id=DISCORD_SERVER_ID),
)
async def backfill(interaction):
    view = OnboardingView()
    await interaction.response.send_message("Click the button below:", view=view, ephemeral=False)

@tree.command(
        name="goals",
        description="To get your active goals",
        guild=discord.Object(id=DISCORD_SERVER_ID),
)
async def user_goals(interaction):
    await interaction.response.defer(ephemeral=False, thinking=True)

    goals = await get_user_goals(interaction.user.id)
    if goals:
        msg_parts = [f"{goal.goal_description}: {goal.target} {goal.metric}, {goal.frequency} times a week" for goal in goals]
        goals_message = "\n".join(msg_parts)
        await interaction.followup.send(f"Here are your current goals:\n{goals_message}", ephemeral=False)
    else:
        await interaction.followup.send("You currently have no active goals.", ephemeral=False)

@tasks.loop(hours=24)
async def send_weekly_leaderboard():
    # TODO: run every hour, but send only after 24 hours from the last report
    # Leaderboard task only runs in production on Heroku
    logging.info('Sending weekly leaderboard')
    channel = await client.fetch_channel(GENERAL_CHANNEL_ID)
    #Get last message in channel
    async for last_message in channel.history(limit=1):
        if last_message and last_message.author == client.user:
            if "Weekly leaderboard" in last_message.content:
                time_since_last_message = datetime.now(timezone.utc) - last_message.created_at
                if time_since_last_message < timedelta(hours=24):
                    # It's been less than 24 hours since the last leaderboard message
                    logging.info("Skipping sending leaderboard as it's been less than 24 hours.")
                    return
                
    leaderboards = await get_weekly_leaderboard()

    current_date = datetime.now().strftime("%d-%b-%Y")

    msg_parts = [f"**Weekly leaderboard: {current_date}**\n"]

    for category, leaderboard in leaderboards.items():
        if leaderboard:
            msg_parts.append(f"**{category}**")
            for user_id, submissions, days in leaderboard:
                username = (await client.fetch_user(user_id)).global_name
                msg_parts.append(f"{username}: {submissions} submissions, {days} active days")
            msg_parts.append("")

    msg = "\n".join(msg_parts)

    await channel.send(msg)
    logging.info('Successfully sent weekly leaderboard')


async def init():
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise Exception("Please set the DATABASE_URL environment variable")

    await init_db(DATABASE_URL)
    
    discord.utils.setup_logging()
    return await client.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(init())
