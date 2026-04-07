const path = require("path");
const ROOT = __dirname;
const PYTHON = path.join(ROOT, ".venv", "Scripts", "python.exe");
const NODE = process.execPath;
const NEXT = path.join(ROOT, "web", "frontend", "node_modules", "next", "dist", "bin", "next");

module.exports = {
  apps: [
    {
      name: "backend",
      cwd: path.join(ROOT, "web", "backend"),
      script: PYTHON,
      args: "-m uvicorn main:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      windowsHide: true,
      max_restarts: 3,
      min_uptime: "10s",
    },
    {
      name: "mt5-worker",
      cwd: path.join(ROOT, "web", "backend"),
      script: PYTHON,
      args: "mt5_worker.py start",
      interpreter: "none",
      windowsHide: true,
      max_restarts: 3,
      min_uptime: "10s",
    },
    {
      name: "tma-worker",
      cwd: path.join(ROOT, "web", "backend"),
      script: PYTHON,
      args: "tma_worker.py start",
      interpreter: "none",
      windowsHide: true,
      max_restarts: 3,
      min_uptime: "10s",
    },
    {
      name: "frontend",
      cwd: path.join(ROOT, "web", "frontend"),
      script: NEXT,
      args: "dev -p 3000",
      interpreter: NODE,
      windowsHide: true,
      max_restarts: 3,
      min_uptime: "10s",
    },
  ],
};
