import pytest
from datetime import datetime, timedelta
from src.submissions.llm_submissions import _process_csv_submission, ParsedSubmissionItem

@pytest.mark.asyncio
async def test_process_csv_submission():
    # Test Data
    csv_data = "0, Fitness, 30\n-1, Reading, true"
    # Mapping of category names to goal IDs [done in parse_submission_message()]
    category_name_to_goal_id = {"Fitness": 1, "Reading": 2}
    
    # Expected Submission
    expected = [
        ParsedSubmissionItem(category="Fitness", goal_id=1, value=30, submission_time=datetime.now()),
        ParsedSubmissionItem(category="Reading", goal_id=2, value=None, submission_time=(datetime.now() - timedelta(days=1)))
    ]

    # Call the function to test
    result = _process_csv_submission(csv_data, category_name_to_goal_id)

    # Verify each item 
    for expected_item, result_item in zip(expected, result):
        assert expected_item.category == result_item.category
        assert expected_item.goal_id == result_item.goal_id
        assert expected_item.value == result_item.value
        # Ensure dates ignoring time
        assert expected_item.submission_time.date() == result_item.submission_time.date()
