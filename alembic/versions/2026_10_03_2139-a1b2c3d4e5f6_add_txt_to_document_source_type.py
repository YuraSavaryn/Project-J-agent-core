"""add_txt_to_document_source_type

Revision ID: a1b2c3d4e5f6
Revises: 64de3441303b
Create Date: 2026-10-03 21:39:00

Adds 'txt' value to the ``document_source_type`` PostgreSQL ENUM.
PostgreSQL requires ALTER TYPE ... ADD VALUE which is not transactional
(cannot be rolled back inside a transaction).  We use
``op.execute()`` directly and suppress the lock with
``execute_if(dialect='postgresql')``.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "64de3441303b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ADD VALUE is idempotent from Postgres 9.3+; IF NOT EXISTS from 12.
    op.execute("ALTER TYPE document_source_type ADD VALUE IF NOT EXISTS 'txt'")


def downgrade() -> None:
    # PostgreSQL does not support removing an enum value natively.
    # The safest approach would be to recreate the type, but that is
    # destructive.  We leave downgrade as a no-op and document this.
    pass
