# enable langsmith

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
import json
import time
import logging
from typing import List, Tuple
from langchain_anthropic.experimental import ChatAnthropicTools
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain.schema import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# the SingleSubmission and UserSubmissions objects follows current db schema
class SingleSubmission(BaseModel):
    goal_id: int = Field(..., description="The goal_id that the user is tracking")
    amount: float = Field(..., description="Value of the metric that the user is tracking")
    proof_str: str = Field(..., description="The string statement that the user submitted")

class UsersSubmissions(BaseModel):
    submissions: List[SingleSubmission] = Field(..., description="The list of progress submissions")

class Goal(BaseModel):
    goal_id: int = Field(..., description="The goal_id that the user is tracking")
    category_name: str = Field(..., description="The category of the goal e.g. Fitness, Coding, Studying, Meditation, Content Creation, etc.")
    goal_description: str = Field(..., description="The description of the goal e.g. I want to meditate for 10 minutes every day")
    metric: str = Field(..., description="The metric that the user is tracking e.g. minutes, problems, articles")
    target: int = Field(..., description="The target value of the metric for a day e.g. 3, 5")
    weekly_frequency: int = Field(..., description="The frequency of the metric tracked per week e.g. 1, 2, 3")

class UserGoals(BaseModel):
    # username: str = Field(..., description="The username of the user") # this can be changed to user_id
    goals: list[Goal] = Field(..., description="The list of goals for the user")


def extract_progress(profile: UserGoals, submission: str, model):
    """Extracts the progress from the submission text.

    Args:
        submission (str): The submission text.

    Returns:
        UserProgress: The progress of the user.
    """

    template = """
    Given the user's goals, extract their progress update from their text.

    {profile}

    {submission_text}

    In your response only include the progress submissions that are relevant to the user's goals. and don't include any that were not referenced in the submission text.

    Respond with the following format:

    {format_instructions}

    Ensure that its acceptible json format and that you're not including any exit characters or extra braces
    """

 
    parser = PydanticOutputParser(pydantic_object=UsersSubmissions) #Option to use JsonOutputParser(pydantic_object=UsersSubmissions)
    
    prompt = PromptTemplate(
        input_variables=["goals", "submission_text"],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions() }
    )

    chain = prompt | model | parser

    try:
        output = chain.invoke({"profile": profile.goals, "submission_text": submission})
        return output
    except OutputParserException as e:
        for i in range(3):
            try:
                output = chain.invoke({"profile": profile.goals, "submission_text": submission})
                break
            except OutputParserException as e:
                print(e)
                print(f"Retrying {i+1}/3")
                time.sleep(1)

    return {"error": "Model couldn't process the question"}
