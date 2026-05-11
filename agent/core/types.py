from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class InboundMessage:
    user_id: int
    chat_id: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutboundMessage:
    chat_id: int
    content: str
    format: str = "text"


@dataclass
class MemoryItem:
    id: UUID
    user_id: int
    memory_type: str
    summary: str
    embedding: list[float] | None
    status: str
    source_ref: str | None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    user_id: int
    chat_id: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PipelineContext:
    base_class_marker: bool = field(init=False, default=True)


@dataclass
class BeforeTurnCtx(PipelineContext):
    inbound_message: InboundMessage
    session: Session
    retrieved_memories: list[MemoryItem]


@dataclass
class BeforeReasoningCtx(PipelineContext):
    session: Session
    memories: list[MemoryItem]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


@dataclass
class ReasonerResult:
    content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str


@dataclass
class AfterReasoningCtx(PipelineContext):
    reasoner_result: ReasonerResult
    outbound_message: OutboundMessage


@dataclass
class TurnCommittedEvent:
    turn_id: str
    user_id: int
    outbound_message: OutboundMessage
    new_memory_ids: list[UUID]
    timestamp: datetime = field(default_factory=datetime.utcnow)
