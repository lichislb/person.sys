"""Initialize local SQLite database for MVP events."""

from __future__ import annotations

from src.service.event_store import EventStore


def main() -> None:
    """Create SQLite database and events table."""
    db_path = "data/streamlit_events.db"
    store = EventStore(db_path=db_path)
    store.init_db()
    print("数据库初始化成功")
    print(f"数据库路径: {db_path}")


if __name__ == "__main__":
    main()
