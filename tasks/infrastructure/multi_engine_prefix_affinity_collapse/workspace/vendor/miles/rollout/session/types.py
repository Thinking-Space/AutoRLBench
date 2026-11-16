from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    timestamp: float
    request_timestamp: float | None = None
    method: str
    path: str
    request: dict
    response: dict
    status_code: int
    # Multi-sequence providers tag each turn with the sequence it belongs to.
    # Single-sequence providers leave it unset, so the field serializes as null
    # and existing readers (model_validate / plain dict access) are unaffected.
    sequence_id: str | None = None


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
