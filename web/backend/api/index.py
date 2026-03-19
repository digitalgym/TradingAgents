"""
TradingAgents API for Vercel deployment.
Uses simple HTTP handler - Vercel compatible.
"""
from http.server import BaseHTTPRequestHandler
import json
import os
from urllib.parse import urlparse, parse_qs

# Database
DATABASE_URL = os.environ.get("POSTGRES_URL", "")
API_KEY = os.environ.get("API_KEY", "")

# Convert URL for pg8000 and remove sslmode (pg8000 uses ssl_context instead)
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+pg8000://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+pg8000://", 1)
    # Remove sslmode param - pg8000 handles SSL differently
    if "sslmode=" in DATABASE_URL:
        import re
        DATABASE_URL = re.sub(r'[?&]sslmode=[^&]*', '', DATABASE_URL)
        # Clean up any remaining ? or & issues
        DATABASE_URL = DATABASE_URL.replace('?&', '?').rstrip('?')

_engine = None

def get_engine():
    global _engine
    if _engine is None and DATABASE_URL:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool
        _engine = create_engine(DATABASE_URL, poolclass=NullPool)
    return _engine


def json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    from datetime import datetime, date
    from decimal import Decimal
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


class handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=json_serial).encode())

    def check_api_key(self):
        if not API_KEY:
            return True
        # Headers are case-insensitive but BaseHTTPRequestHandler lowercases them
        headers_lower = {k.lower(): v for k, v in self.headers.items()}
        key = headers_lower.get('x-api-key', '')
        return key == API_KEY

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Health endpoints (no auth)
        if path == '/' or path == '':
            return self.send_json({"status": "ok", "service": "TradingAgents API"})

        if path == '/api/health':
            engine = get_engine()
            db_ok = False
            if engine:
                try:
                    from sqlalchemy import text
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    db_ok = True
                except Exception as e:
                    return self.send_json({"status": "ok", "database": False, "error": str(e)})
            return self.send_json({"status": "ok", "database": db_ok})

        # Protected endpoints
        if not self.check_api_key():
            return self.send_json({"error": "Invalid or missing API key"}, 401)

        # Decisions
        if path == '/api/decisions':
            return self.get_decisions(query)
        if path == '/api/decisions/stats':
            return self.get_decision_stats()
        if path.startswith('/api/decisions/'):
            decision_id = path.split('/')[-1]
            return self.get_decision(decision_id)

        # Trade queue
        if path == '/api/trade-queue':
            return self.get_trade_queue(query)

        # Automation
        if path == '/api/automation/status':
            return self.get_automation_status(query)
        if path == '/api/automation/control/pending':
            return self.get_pending_controls(query)

        return self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if not self.check_api_key():
            return self.send_json({"error": "Invalid or missing API key"}, 401)

        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        if path == '/api/trade-queue':
            return self.queue_trade(body)
        if path == '/api/automation/control':
            return self.automation_control(body)

        return self.send_json({"error": "Not found"}, 404)

    def get_decisions(self, query):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        from sqlalchemy import text
        status = query.get('status', [None])[0]
        symbol = query.get('symbol', [None])[0]
        limit = int(query.get('limit', [100])[0])
        offset = int(query.get('offset', [0])[0])

        conditions = []
        params = {"limit": limit, "offset": offset}

        if status == "active":
            conditions.append("closed_at IS NULL")
        elif status == "closed":
            conditions.append("closed_at IS NOT NULL")
        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM decisions {where} ORDER BY created_at DESC LIMIT :limit OFFSET :offset"

        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = [dict(row._mapping) for row in result]
        return self.send_json({"decisions": rows})

    def get_decision_stats(self):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) as total,
                       COUNT(*) FILTER (WHERE exit_date IS NULL) as active,
                       COUNT(*) FILTER (WHERE exit_date IS NOT NULL) as closed,
                       COUNT(*) FILTER (WHERE (data->>'was_correct')::boolean = true) as wins,
                       COUNT(*) FILTER (WHERE (data->>'was_correct')::boolean = false) as losses,
                       COALESCE(SUM((data->>'pnl')::numeric) FILTER (WHERE exit_date IS NOT NULL), 0) as total_pnl
                FROM decisions
            """))
            overall = dict(result.fetchone()._mapping)

            result = conn.execute(text("""
                SELECT symbol, COUNT(*) as total,
                       COUNT(*) FILTER (WHERE (data->>'was_correct')::boolean = true) as wins,
                       COUNT(*) FILTER (WHERE (data->>'was_correct')::boolean = false) as losses,
                       COALESCE(SUM((data->>'pnl')::numeric), 0) as total_pnl
                FROM decisions GROUP BY symbol ORDER BY COUNT(*) DESC
            """))
            by_symbol = [dict(row._mapping) for row in result]

        return self.send_json({
            "total": overall["total"],
            "active": overall["active"],
            "closed": overall["closed"],
            "wins": overall["wins"],
            "losses": overall["losses"],
            "total_pnl": float(overall["total_pnl"]),
            "win_rate": round(overall["wins"] / overall["closed"] * 100, 1) if overall["closed"] else 0,
            "by_symbol": {r["symbol"]: {"total": r["total"], "wins": r["wins"], "losses": r["losses"], "total_pnl": float(r["total_pnl"])} for r in by_symbol}
        })

    def get_decision(self, decision_id):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM decisions WHERE id = :id"), {"id": decision_id})
            row = result.fetchone()
            if not row:
                return self.send_json({"error": "Not found"}, 404)
        return self.send_json(dict(row._mapping))

    def get_trade_queue(self, query):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        from sqlalchemy import text
        status = query.get('status', [None])[0]

        with engine.connect() as conn:
            if status:
                result = conn.execute(text("SELECT * FROM trade_queue WHERE status = :status ORDER BY created_at DESC LIMIT 100"), {"status": status})
            else:
                result = conn.execute(text("SELECT * FROM trade_queue ORDER BY created_at DESC LIMIT 100"))
            rows = [dict(row._mapping) for row in result]
        return self.send_json({"items": rows})

    def queue_trade(self, body):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        command = body.get("command")
        if not command:
            return self.send_json({"error": "command required"}, 400)

        payload = {}
        if command == "execute":
            payload = {"symbol": body.get("symbol"), "direction": body.get("direction"), "volume": body.get("volume"),
                       "entry": body.get("entry"), "sl": body.get("sl"), "tp": body.get("tp")}
        elif command == "close":
            payload = {"ticket": body.get("ticket")}
        elif command in ("modify_sl", "modify_tp"):
            payload = {"ticket": body.get("ticket"), "new_sl": body.get("new_sl"), "new_tp": body.get("new_tp")}

        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO trade_queue (command, payload, status, created_at)
                VALUES (:command, :payload, 'pending', NOW()) RETURNING id
            """), {"command": command, "payload": json.dumps(payload)})
            row = result.fetchone()
            conn.commit()

        return self.send_json({"success": True, "id": str(row[0]), "message": f"Queued: {command}"})

    def get_automation_status(self, query):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        from sqlalchemy import text
        instance = query.get('instance', [None])[0]

        with engine.connect() as conn:
            if instance:
                result = conn.execute(text("SELECT * FROM automation_status WHERE instance_name = :instance"), {"instance": instance})
                row = result.fetchone()
                return self.send_json({"status": dict(row._mapping) if row else None})
            else:
                result = conn.execute(text("SELECT * FROM automation_status ORDER BY instance_name"))
                rows = [dict(row._mapping) for row in result]
        return self.send_json({"statuses": rows})

    def automation_control(self, body):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        instance = body.get("instance")
        action = body.get("action")
        config = body.get("config")

        if not instance or not action:
            return self.send_json({"error": "instance and action required"}, 400)

        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO automation_control (instance_name, action, config, status, created_at)
                VALUES (:instance, :action, :config, 'pending', NOW()) RETURNING id
            """), {"instance": instance, "action": action, "config": json.dumps(config) if config else None})
            row = result.fetchone()
            conn.commit()

        return self.send_json({"success": True, "id": str(row[0]), "message": f"Queued: {action} for {instance}"})

    def get_pending_controls(self, query):
        engine = get_engine()
        if not engine:
            return self.send_json({"error": "Database not configured"}, 503)

        from sqlalchemy import text
        instance = query.get('instance', [None])[0]

        with engine.connect() as conn:
            if instance:
                result = conn.execute(text("SELECT * FROM automation_control WHERE instance_name = :instance AND status = 'pending' ORDER BY created_at"), {"instance": instance})
            else:
                result = conn.execute(text("SELECT * FROM automation_control WHERE status = 'pending' ORDER BY created_at"))
            rows = [dict(row._mapping) for row in result]
        return self.send_json({"commands": rows})
