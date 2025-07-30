import asyncio
from fails.pipeline import run_extract_and_classify_pipeline


USER_AI_SYSTEM_CONTEXT = """My app is trying to idenify insights from transcripts of \
meetings between prospects and our sales team."""

USER_EVAL_CONTEXT = """To classify speaker IDs from a transcript into whether they \
are from our company or are a customer/prospect."""

user_context_str = f"""
## User AI System Context

What the user is trying to achieve with their AI system: 

<user_ai_system_context>
{USER_AI_SYSTEM_CONTEXT}
</user_ai_system_context>

## User Eval Context 

What the user is trying to evaluate in their AI system: 

<user_eval_context>
{USER_EVAL_CONTEXT}
</user_eval_context>

"""

asyncio.run(
    run_extract_and_classify_pipeline(
        eval_id="0197a72d-2704-7ced-8c07-0fa1e0ab0557",
        wandb_entity="wandb-applied-ai-team",
        wandb_project="eval-failures",
        user_context=user_context_str,
        config_file_path="config/speaker_classification_eval_config.yaml",  # sets the columns to filter by
        model="gemini/gemini-2.5-flash",
        force_column_selection=False,
        debug=True,
        max_concurrent_llm_calls=100,
    )
)