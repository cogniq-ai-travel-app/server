import os
import time
import tempfile
import threading
import requests

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("EXPO_PUBLIC_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("EXPO_PUBLIC_SUPABASE_ANON_KEY")
PING_INTERVAL_HOURS = 24
PING_FILE = os.path.join(tempfile.gettempdir(), "supabase_last_ping.txt")

def _perform_ping():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("[SUPABASE KEEPALIVE] Skipping ping: Credentials not found in environment.", flush=True)
        return

    print(f"[SUPABASE KEEPALIVE] Triggering wake-up ping to Supabase: {SUPABASE_URL}", flush=True)
    try:
        # 1. Ping REST API root (requires api key) to keep PostgREST and PostgreSQL warm
        rest_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/"
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
        }
        res_rest = requests.get(rest_url, headers=headers, timeout=12)
        print(f"[SUPABASE KEEPALIVE] REST API response: status={res_rest.status_code}", flush=True)

        # 2. Ping Auth health endpoint to keep GoTrue/Auth container warm
        auth_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/health"
        res_auth = requests.get(auth_url, timeout=12)
        print(f"[SUPABASE KEEPALIVE] Auth API response: status={res_auth.status_code}", flush=True)

        # Update last ping timestamp file on successful execution
        try:
            with open(PING_FILE, "w") as f:
                f.write(str(int(time.time())))
            print(f"[SUPABASE KEEPALIVE] Updated last ping timestamp in: {PING_FILE}", flush=True)
        except Exception as fe:
            print(f"[SUPABASE KEEPALIVE] Error writing timestamp: {fe}", flush=True)

    except Exception as e:
        print(f"[SUPABASE KEEPALIVE] Keepalive ping failed: {e}", flush=True)

def check_and_ping():
    """
    Checks if the last ping was more than 24 hours ago.
    Spawns a non-blocking daemon thread if a ping is needed.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return

    now = int(time.time())
    last_ping = 0

    if os.path.exists(PING_FILE):
        try:
            with open(PING_FILE, "r") as f:
                last_ping = int(f.read().strip())
        except Exception:
            pass

    # If last check was more than 24 hours ago (or never check)
    if now - last_ping > PING_INTERVAL_HOURS * 3600:
        # Spawn keepalive ping asynchronously in the background
        threading.Thread(target=_perform_ping, daemon=True).start()
