#!/usr/bin/env bash
# backend/scripts/init_postgres.sh
# ---------------------------------
# Idempotent setup for the SOD task-history database.
#
# - Creates a dedicated app role (default: sod_app) with a strong random
#   password, or reuses the password already in backend/.env if present.
# - Creates the application database (default: sod) owned by that role.
# - Grants schema privileges and verifies a password login over TCP.
# - Writes / updates DATABASE_URL in backend/.env.
#
# Requires sudo (uses peer auth to act as the local 'postgres' OS user).
#
# Usage:
#   bash backend/scripts/init_postgres.sh        # if `sudo -v` is already cached
#   sudo bash backend/scripts/init_postgres.sh   # otherwise — prompts for sudo pw

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$HERE/.." && pwd)"
ENV_FILE="$BACKEND_DIR/.env"
ENV_EXAMPLE="$BACKEND_DIR/.env.example"

DB_NAME="${SOD_DB_NAME:-sod}"
DB_USER="${SOD_DB_USER:-sod_app}"
DB_HOST="${SOD_DB_HOST:-localhost}"
DB_PORT="${SOD_DB_PORT:-5432}"

# ── sanity ─────────────────────────────────────────────────────────────────
command -v psql    >/dev/null || { echo "✗ psql not installed"; exit 1; }
command -v openssl >/dev/null || { echo "✗ openssl not installed"; exit 1; }
command -v python3 >/dev/null || { echo "✗ python3 not installed"; exit 1; }
pg_isready -q || { echo "✗ PostgreSQL is not accepting connections on $DB_HOST:$DB_PORT"; exit 1; }

# Admin DDL runs as the postgres OS user (peer auth on local socket).
PSQL_ADMIN=(sudo -u postgres psql -v ON_ERROR_STOP=1 -X -A -t)

# ── reuse password from .env if already present (rerun friendly) ────────────
DB_PASS=""
if [[ -f "$ENV_FILE" ]]; then
  EXISTING="$(grep -E '^DATABASE_URL=' "$ENV_FILE" | head -1 || true)"
  if [[ "$EXISTING" =~ ://[^:]+:([^@]+)@ ]]; then
    DB_PASS="${BASH_REMATCH[1]}"
  fi
fi
if [[ -z "$DB_PASS" ]]; then
  # 24 chars of [A-Za-z0-9] — safe in a URL, plenty of entropy for a local role.
  DB_PASS="$(openssl rand -base64 32 | tr -dc 'A-Za-z0-9' | head -c 24)"
fi

# ── role ────────────────────────────────────────────────────────────────────
echo "▶ Ensuring role '$DB_USER'…"
"${PSQL_ADMIN[@]}" <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER') THEN
    CREATE ROLE "$DB_USER" WITH LOGIN PASSWORD '$DB_PASS';
  ELSE
    ALTER ROLE "$DB_USER" WITH LOGIN PASSWORD '$DB_PASS';
  END IF;
END
\$\$;
SQL

# ── database ───────────────────────────────────────────────────────────────
echo "▶ Ensuring database '$DB_NAME'…"
EXISTS="$("${PSQL_ADMIN[@]}" -c "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" || true)"
if [[ "$EXISTS" != "1" ]]; then
  "${PSQL_ADMIN[@]}" -c "CREATE DATABASE \"$DB_NAME\" OWNER \"$DB_USER\";"
fi

# ── grants (idempotent) ────────────────────────────────────────────────────
"${PSQL_ADMIN[@]}" -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO \"$DB_USER\";" >/dev/null

# ── verify TCP login (the path the app actually uses) ─────────────────────
echo "▶ Verifying password login over TCP…"
PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
  "SELECT 'connected as ' || current_user || ' to ' || current_database();"

# ── update .env ────────────────────────────────────────────────────────────
NEW_URL="postgresql+asyncpg://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME"

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$ENV_EXAMPLE" ]]; then cp "$ENV_EXAMPLE" "$ENV_FILE"; else : > "$ENV_FILE"; fi
fi

if grep -qE '^DATABASE_URL=' "$ENV_FILE"; then
  python3 - "$ENV_FILE" "$NEW_URL" <<'PY'
import re, sys
path, new = sys.argv[1], sys.argv[2]
with open(path, 'r', encoding='utf-8') as f: text = f.read()
text = re.sub(r'^DATABASE_URL=.*$', 'DATABASE_URL=' + new, text, count=1, flags=re.M)
with open(path, 'w', encoding='utf-8') as f: f.write(text)
PY
else
  printf '\n# Auto-added by init_postgres.sh\nDATABASE_URL=%s\n' "$NEW_URL" >> "$ENV_FILE"
fi

# .env now contains a password — tighten perms.
chmod 600 "$ENV_FILE" 2>/dev/null || true

echo
echo "✓ All done."
masked="$(printf '%s' "$NEW_URL" | sed -E 's|(://)([^:]+):([^@]+)(@.*)|\1\2:****\4|')"
echo "  DATABASE_URL written to $ENV_FILE"
echo "  Effective : $masked"
echo
echo "Next:"
echo "  pip install -r backend/requirements.txt"
echo "  bash backend/start.sh   # init_db() will create the detection_tasks table"
