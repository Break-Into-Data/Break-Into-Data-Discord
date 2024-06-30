import asyncio
import logging
import discord

from src.database import (
    create_external_platform_connection,
    get_external_platform,
    new_goal,
    ensure_user, 
    get_category_by_name,
    save_user_personal_details,
)
from src.buttons import is_user_activated
from src.metrics_collection.events import save_event
from src.models import EventType


logger = logging.getLogger(__name__)


LEETCODE_CATEGORY_NAME = "_automated_LeetCode"


class ConnectExternalPlatformModal(discord.ui.Modal):
    def __init__(self, platform_name: str, is_onboarding_required: bool):
        super().__init__(title=f"Connect {platform_name}")
        self.is_onboarding_required = is_onboarding_required
        self.platform_name = platform_name
                
        self.user_name = discord.ui.TextInput(
            label="Your Name",
            placeholder="Enter your name",
            required=True,
        )
        self.email = discord.ui.TextInput(
            label="Email Address",
            placeholder="Enter your email",
            required=True,
        )
        
        self.user_name_input = discord.ui.TextInput(
            label=f"{platform_name} User Name", required=True, placeholder=f"User name for {platform_name}"
        )
        
        if is_onboarding_required:
            self.add_item(self.user_name)
            self.add_item(self.email)
            
        self.add_item(self.user_name_input)

    async def on_submit(self, interaction: discord.Interaction):
        user, category = await asyncio.gather(
            ensure_user(interaction.user),
            get_category_by_name(LEETCODE_CATEGORY_NAME),
        )
        
        if self.is_onboarding_required:
            await save_user_personal_details(
                interaction.user, 
                self.email.value, 
                self.user_name.value
            )

        if category is None:
            await interaction.followup.send(
                "Something went wrong, please contact support", ephemeral=True
            )
            return

        await new_goal(
            user_id=user.user_id,
            category_id=category.category_id,
            goal_description="",
            metric="",
            target=1,
            frequency="",
        )
        
        platform = await get_external_platform(self.platform_name)
        
        await create_external_platform_connection(
            user_id=user.user_id,
            platform_id=platform.platform_id,
            user_name=self.user_name_input.value,
            user_data={},
        )

        await interaction.followup.send(f"{self.platform_name} is connected!", ephemeral=True)
        logger.info(f"External platform {self.platform_name} connected for user {interaction.user}")


class ConnectExternalPlatform(discord.ui.View):
    """ Creates a view with the buttons for connecting 
    services like LeetCode, GitHub, etc.
    """

    def __init__(self):
        super().__init__(timeout=None)
        
        self.add_item(discord.ui.Button(
            label="Connect LeetCode", style=discord.ButtonStyle.primary,
            custom_id="LeetCode",
            row=0,
        ))
    
    async def interaction_check(self, interaction: discord.Interaction):
        if interaction.type != discord.InteractionType.component:
            return True
        
        user = await ensure_user(interaction.user)
        
        is_activated = is_user_activated(user)
        modal = ConnectExternalPlatformModal(
            platform_name=interaction.data.get("custom_id"),
            is_onboarding_required=not is_activated
        )
        await interaction.response.send_modal(modal)
        
        return False
    