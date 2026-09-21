"""Pydantic data model for a single normalized Qudurat question."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, field_validator

Difficulty = Literal["easy", "medium", "hard"]
Status = Literal["pending_review", "approved", "rejected"]

CONFIDENCE_APPROVE_THRESHOLD = 0.95


class QuestionOptions(BaseModel):
    A: str = ""
    B: str = ""
    C: str = ""
    D: str = ""


class QudurateQuestion(BaseModel):
    """A single extracted, normalized, self-verified Qudurat question."""

    source_page: int
    source_pdf: str = ""
    box_index: Optional[int] = None
    skill: str
    sub_skill: str
    difficulty: Difficulty
    question_text: str
    options: QuestionOptions
    correct_answer: Optional[str] = None
    explanation: str = ""
    mathematically_verified: bool = False
    confidence_score: float = 0.0
    status: Status = "pending_review"
    has_image: bool = False
    image_url: Optional[str] = None  # set after extraction, only when has_image is true

    @field_validator("correct_answer")
    @classmethod
    def _normalize_answer(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        if v == "" or v == "NULL":
            return None
        if v not in {"A", "B", "C", "D"}:
            raise ValueError(f"correct_answer must be one of A/B/C/D or null, got {v!r}")
        return v

    @field_validator("confidence_score")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def compute_status(self) -> Status:
        """Approve only when the model both solved the equation itself and is highly
        confident; everything else needs a human look in the review dashboard."""
        if self.mathematically_verified and self.confidence_score >= CONFIDENCE_APPROVE_THRESHOLD:
            return "approved"
        return "pending_review"

    def embedding_text(self) -> str:
        """Text combining the question and its options, used for embeddings."""
        options_text = " ".join(
            f"{key}) {value}" for key, value in self.options.model_dump().items() if value
        )
        return f"{self.question_text}\n{options_text}".strip()

    def to_supabase_row(self, embedding: list[float]) -> dict:
        return {
            "source_page": self.source_page,
            "source_pdf": self.source_pdf,
            "box_index": self.box_index,
            "skill": self.skill,
            "sub_skill": self.sub_skill,
            "difficulty": self.difficulty,
            "question_text": self.question_text,
            "options": self.options.model_dump(),
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "mathematically_verified": self.mathematically_verified,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "has_image": self.has_image,
            "image_url": self.image_url,
            "embedding": embedding,
        }


# JSON Schema handed to the OpenAI Vision model for strict structured output.
# One box/page can contain zero, one, or several questions. "solving_work" is
# placed before the verification fields so the model's own algebraic work
# (required by the self-correction protocol) is generated first and can
# actually influence the mathematically_verified/confidence_score/correct_answer
# tokens that follow, even within this single structured-output call.
PAGE_EXTRACTION_JSON_SCHEMA = {
    "name": "qudurat_page_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "sub_skill": {"type": "string"},
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                        },
                        "question_text": {"type": "string"},
                        "options": {
                            "type": "object",
                            "properties": {
                                "A": {"type": "string"},
                                "B": {"type": "string"},
                                "C": {"type": "string"},
                                "D": {"type": "string"},
                            },
                            "required": ["A", "B", "C", "D"],
                            "additionalProperties": False,
                        },
                        "solving_work": {"type": "string"},
                        "mathematically_verified": {"type": "boolean"},
                        "confidence_score": {"type": "number"},
                        "correct_answer": {
                            "type": ["string", "null"],
                            "enum": ["A", "B", "C", "D", None],
                        },
                        "explanation": {"type": "string"},
                        "has_image": {"type": "boolean"},
                    },
                    "required": [
                        "skill",
                        "sub_skill",
                        "difficulty",
                        "question_text",
                        "options",
                        "solving_work",
                        "mathematically_verified",
                        "confidence_score",
                        "correct_answer",
                        "explanation",
                        "has_image",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}
