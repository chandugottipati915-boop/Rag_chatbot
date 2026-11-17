"""All chat-history queries in one place, so routes stay short and tests can reuse them."""
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag.db.models import Conversation, Message, utcnow


def list_conversations(db: Session, user_id: str | None = None, limit: int = 50) -> list[Conversation]:
    stmt = (select(Conversation)
            .where(Conversation.user_id == user_id if user_id else Conversation.user_id.is_(None))
            .order_by(Conversation.updated_at.desc())
            .limit(limit))
    return list(db.scalars(stmt))


def create_conversation(db: Session, title: str, user_id: str | None = None) -> Conversation:
    conv = Conversation(title=title[:200], user_id=user_id)
    db.add(conv)
    db.commit()
    return conv


def get_conversation(db: Session, conversation_id: uuid.UUID) -> Conversation | None:
    return db.get(Conversation, conversation_id)


def rename_conversation(db: Session, conv: Conversation, title: str) -> Conversation:
    conv.title = title[:200]
    conv.updated_at = utcnow()
    db.commit()
    return conv


def delete_conversation(db: Session, conv: Conversation) -> None:
    db.delete(conv)          # cascade="all, delete-orphan" deletes its messages too
    db.commit()


def recent_history(db: Session, conversation_id: uuid.UUID, limit: int = 20) -> list[dict]:
    """The last `limit` messages, oldest first, in the shape the graph expects."""
    stmt = (select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit))
    newest_first = list(db.scalars(stmt))
    return [{"role": m.role, "content": m.content} for m in reversed(newest_first) if m.content]


def add_message(db: Session, conversation_id: uuid.UUID, role: str, content: str,
                sources: list | None = None, meta: dict | None = None) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content,
                  sources=sources, meta=meta)
    db.add(msg)
    conv = db.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = utcnow()          # so the chat moves to the top of the list
    db.commit()
    return msg
