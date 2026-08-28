"""
app/rate_limiter.py
─────────────────────────────────────────────────────────────────────────────
The Limiter instance lives here, separately from main.py, because both
main.py (to attach it to app.state) and routes.py (to decorate /predict)
need it — importing it from main.py would create main -> routes -> main.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
