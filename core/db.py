"""SQLite persistence: users, datasets, runs, anomaly reviews, settings, audit log."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('analyst','admin')),
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  features_path TEXT NOT NULL,
  labels_path TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  n_rows INTEGER, n_features INTEGER,
  created_by TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  dataset_id INTEGER NOT NULL,
  config_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  progress REAL NOT NULL DEFAULT 0,
  history_json TEXT NOT NULL DEFAULT '{}',
  metrics_json TEXT NOT NULL DEFAULT '{}',
  model_dir TEXT,
  published INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  created_by TEXT, created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS reviews(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL, record_id TEXT NOT NULL,
  score REAL, verdict TEXT NOT NULL CHECK(verdict IN ('true_positive','false_positive')),
  reviewer TEXT, created_at TEXT NOT NULL,
  UNIQUE(run_id, record_id)
);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, username TEXT, action TEXT NOT NULL, detail TEXT
);
"""

DEFAULT_SETTINGS = {"flag_percentile": "5"}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.executescript(SCHEMA)
        for k, v in DEFAULT_SETTINGS.items():
            c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))


def q(sql: str, args: tuple = ()) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def q1(sql: str, args: tuple = ()) -> dict | None:
    rows = q(sql, args)
    return rows[0] if rows else None


def ex(sql: str, args: tuple = ()) -> int:
    with conn() as c:
        return c.execute(sql, args).lastrowid


# ---- audit / settings -------------------------------------------------------
def log(username: str | None, action: str, detail: str = "") -> None:
    ex("INSERT INTO audit(ts,username,action,detail) VALUES(?,?,?,?)", (now(), username, action, detail))


def get_setting(key: str) -> str:
    r = q1("SELECT value FROM settings WHERE key=?", (key,))
    return r["value"] if r else DEFAULT_SETTINGS[key]


def set_setting(key: str, value: str) -> None:
    ex("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


# ---- datasets ---------------------------------------------------------------
def add_dataset(name, features_path, labels_path, meta: dict, n_rows, n_features, created_by) -> int:
    return ex(
        "INSERT INTO datasets(name,features_path,labels_path,meta_json,n_rows,n_features,created_by,created_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (name, str(features_path), str(labels_path) if labels_path else None, json.dumps(meta),
         n_rows, n_features, created_by, now()),
    )


def list_datasets() -> list[dict]:
    return q("SELECT * FROM datasets ORDER BY id")


def get_dataset(dataset_id: int) -> dict | None:
    d = q1("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if d:
        d["meta"] = json.loads(d["meta_json"])
    return d


# ---- runs -------------------------------------------------------------------
def create_run(name, dataset_id, config: dict, created_by) -> int:
    return ex(
        "INSERT INTO runs(name,dataset_id,config_json,created_by,created_at) VALUES(?,?,?,?,?)",
        (name, dataset_id, json.dumps(config), created_by, now()),
    )


def get_run(run_id: int) -> dict | None:
    r = q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if r:
        r["config"] = json.loads(r["config_json"])
        r["history"] = json.loads(r["history_json"] or "{}")
        r["metrics"] = json.loads(r["metrics_json"] or "{}")
    return r


def list_runs(where: str = "", args: tuple = ()) -> list[dict]:
    rows = q(f"SELECT * FROM runs {where} ORDER BY id DESC", args)
    for r in rows:
        r["config"] = json.loads(r["config_json"])
        r["history"] = json.loads(r["history_json"] or "{}")
        r["metrics"] = json.loads(r["metrics_json"] or "{}")
    return rows


def update_run(run_id: int, **fields) -> None:
    for k in ("history", "metrics"):
        if k in fields:
            fields[f"{k}_json"] = json.dumps(fields.pop(k))
    cols = ", ".join(f"{k}=?" for k in fields)
    ex(f"UPDATE runs SET {cols} WHERE id=?", (*fields.values(), run_id))
