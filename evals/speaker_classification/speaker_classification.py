import weave
import csv
import asyncio
from typing import List, Dict
import os
from pydantic import BaseModel
import litellm

from dotenv import load_dotenv
load_dotenv()


class SpeakerAffiliationResponse(BaseModel):
    speaker_id: str
    reasoning: str
    affiliation: str

SYSTEM_PROMPT = "You are an expert at analyzing conversation transcripts and identifying if speakers are Weights & Biases employees (internal) or external participants (prospects/users). IMPORTANT: By default, classify speakers as external unless there is strong evidence they are internal W&B employees. Technical knowledge alone is not sufficient to classify someone as internal."

DUMB_SYSTEM_PROMPT = """You are an expert at analyzing conversation transcripts and \
identifying if speakers are Weights & Biases employees (internal) or external participants \
(prospects/users).
"""
SYSTEM_PROMPT = DUMB_SYSTEM_PROMPT

USER_PROMPT = "Analyze if speaker {speaker_id} is internal (W&B employee) or external (prospect/user). FULL CONVERSATION CONTEXT: {full_transcript} SPEAKER'S SPECIFIC LINES: {transcript} Remember: Default to external classification unless there is clear evidence the speaker is a W&B employee."
LLM_MODEL = "anthropic/claude-3-5-sonnet-20241022"
LLM_MODEL = "openai/gpt-4.1-mini"

def filter_transcript_by_speaker(full_transcript: str, speaker_id: str) -> str:
    blocks = full_transcript.split("speaker:")
    speaker_texts = []
    for block in blocks:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        if not lines: continue
        if str(speaker_id) in lines[0]:
            text_lines = [line.strip() for line in lines[2:] if line.strip() and not line.strip().startswith('timestamp:')]
            if text_lines:
                speaker_texts.append(' '.join(text_lines))
    return ' '.join(speaker_texts)

class SpeakerAffiliationModel(weave.Model):
    @weave.op()
    def predict(self, speaker_id: str, full_transcript: str, **kwargs) -> dict:
        transcript = filter_transcript_by_speaker(full_transcript, speaker_id)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(speaker_id=speaker_id, transcript=transcript, full_transcript=full_transcript)}
        ]
        response = litellm.completion(
            model=LLM_MODEL,
            messages=messages,
            response_format=SpeakerAffiliationResponse
        )
        content = None
        choices = getattr(response, 'choices', None)
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            message = getattr(choice0, 'message', None)
            content = getattr(message, 'content', None)
        if not content:
            content = getattr(response, 'content', None)
        if not content:
            content = str(response)
        try:
            result = SpeakerAffiliationResponse.model_validate_json(content)
            return {"affiliation": result.affiliation, "reasoning": result.reasoning, "speaker_id": speaker_id}
        except Exception:
            return {"affiliation": "external", "reasoning": f"Could not parse model output: {content}", "speaker_id": speaker_id}

def load_csv_dataset(path: str) -> List[Dict]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

@weave.op()
def affiliation_score(output: dict, affiliation: str) -> dict:
    return {"correct": output["affiliation"].strip().lower() == affiliation.strip().lower()}

def main():
    os.environ["WEAVE_PARALLELISM"] = "10"
    weave.init("wandb-applied-ai-team/eval-failures")
    dataset = weave.ref("speaker_classification") #.get().to_pandas().to_dict(orient="records")
    # dataset = dataset[:2]
    model = SpeakerAffiliationModel()
    evaluation = weave.Evaluation(
        name="speaker_affiliation_eval",
        dataset=dataset, # type: ignore
        scorers=[affiliation_score],
    )
    print(asyncio.run(evaluation.evaluate(model)))

if __name__ == "__main__":
    main()
