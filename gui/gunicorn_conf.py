"""Gunicorn production config for AutoEIS GUI.

Single worker is intentional: task_manager stores running jobs in an
in-process dict; multiple workers would each have their own isolated copy
and jobs would appear to vanish on every other poll request.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8050')}"

# ── Worker model ──────────────────────────────────────────────────────────────
workers = 1            # single process keeps task_manager state consistent
threads = 8            # serve concurrent HTTP requests within that process
worker_class = "gthread"

# ── Timeouts ──────────────────────────────────────────────────────────────────
# Gunicorn kills a worker if a *request* takes longer than `timeout` seconds.
# Analysis runs in a background *thread* so individual HTTP requests (poll,
# btn-click) are always short.  120 s is plenty.
timeout = 120
keepalive = 5
graceful_timeout = 30

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = "-"   # stdout
errorlog  = "-"   # stderr
loglevel  = os.environ.get("LOG_LEVEL", "info")

# ── Security ──────────────────────────────────────────────────────────────────
limit_request_line   = 8190
limit_request_fields = 200
