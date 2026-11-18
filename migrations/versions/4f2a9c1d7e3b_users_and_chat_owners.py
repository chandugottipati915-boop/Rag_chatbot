"""users table; conversations.user_id becomes a foreign key to it

Revision ID: 4f2a9c1d7e3b
Revises: 18b39bed8d42
Create Date: 2026-11-18 10:00:00

Chats saved before auth have no owner (user_id NULL). They stay in the table but no
one can open them; delete them with:  DELETE FROM conversations WHERE user_id IS NULL;
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "4f2a9c1d7e3b"
down_revision: Union[str, Sequence[str], None] = "18b39bed8d42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    # user_id was a placeholder text column (always NULL so far): make it a UUID owner
    op.alter_column("conversations", "user_id",
                    existing_type=sa.String(length=64), type_=sa.Uuid(),
                    postgresql_using="user_id::uuid", existing_nullable=True)
    op.create_foreign_key("conversations_user_id_fkey", "conversations", "users",
                          ["user_id"], ["id"], ondelete="CASCADE")


def downgrade() -> None:
    op.drop_constraint("conversations_user_id_fkey", "conversations", type_="foreignkey")
    op.alter_column("conversations", "user_id",
                    existing_type=sa.Uuid(), type_=sa.String(length=64),
                    postgresql_using="user_id::text", existing_nullable=True)
    op.drop_table("users")
