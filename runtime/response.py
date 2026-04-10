from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VoxMindResponse:
    think_token_ids: List[int]
    answer_token_ids: List[int]
    raw_text: str
    text: str
    think: Optional[str]
    answer: str
    response_prefix: str
    audio_tokens: List[int]
