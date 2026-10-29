import re
from pathlib import Path

from research0.migrate import FORUM_TABLES, MIGRATIONS, TABLES

# Tables that are deliberately not copied by `research0 migrate`. Empty today: a new table must either be listed in migrate.TABLES or be named here with the reason.
NOT_MIGRATED = {}


def created_tables():
    names = set()
    for path in sorted(Path(MIGRATIONS).glob("*.sql")):
        names |= set(re.findall(r"CREATE TABLE(?: IF NOT EXISTS)?\s+([a-z_0-9]+)", path.read_text(encoding="utf-8"), re.I))
    return names


def test_every_table_created_by_a_migration_is_copied_or_excluded_on_purpose():
    covered = {name for name, _, _ in TABLES}
    assert created_tables() - covered - set(NOT_MIGRATED) == set(), "add the table to migrate.TABLES (parents before children) or to NOT_MIGRATED with the reason"


def test_forum_tables_are_all_planned_tables():
    assert FORUM_TABLES <= {name for name, _, _ in TABLES}
