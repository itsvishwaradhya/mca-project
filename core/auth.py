"""Password hashing and user management."""
import bcrypt

from . import db

MIN_PASSWORD_LEN = 8


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode()[:72], hashed.encode())
    except ValueError:
        return False


def create_user(username: str, password: str, role: str = "analyst") -> tuple[bool, str]:
    username = username.strip()
    if not username or " " in username:
        return False, "Username must be non-empty and contain no spaces."
    if len(password) < MIN_PASSWORD_LEN:
        return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if role not in ("analyst", "admin"):
        return False, "Invalid role."
    if db.q1("SELECT 1 FROM users WHERE username=?", (username,)):
        return False, "Username already exists."
    db.ex(
        "INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
        (username, hash_password(password), role, db.now()),
    )
    return True, "User created."


def authenticate(username: str, password: str) -> dict | None:
    u = db.q1("SELECT * FROM users WHERE username=?", (username.strip(),))
    if u and u["active"] and verify_password(password, u["password_hash"]):
        return {"id": u["id"], "username": u["username"], "role": u["role"]}
    return None
