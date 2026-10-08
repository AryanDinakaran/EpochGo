import asyncio
import json
from typing import Dict, List, Set
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        # Maps project_id to a set of active WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Global listeners (e.g. projects list)
        self.global_connections: Set[WebSocket] = set()

    async def connect_project(self, project_id: str, websocket: WebSocket):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()
        self.active_connections[project_id].add(websocket)

    def disconnect_project(self, project_id: str, websocket: WebSocket):
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]

    async def connect_global(self, websocket: WebSocket):
        await websocket.accept()
        self.global_connections.add(websocket)

    def disconnect_global(self, websocket: WebSocket):
        self.global_connections.discard(websocket)

    async def broadcast_to_project(self, project_id: str, message: dict):
        if project_id in self.active_connections:
            dead_connections = set()
            payload = json.dumps(message)
            for connection in self.active_connections[project_id]:
                try:
                    await connection.send_text(payload)
                except Exception:
                    dead_connections.add(connection)
            for dead in dead_connections:
                self.active_connections[project_id].discard(dead)

    async def broadcast_global(self, message: dict):
        dead_connections = set()
        payload = json.dumps(message)
        for connection in self.global_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead_connections.add(connection)
        for dead in dead_connections:
            self.global_connections.discard(dead)


ws_manager = ConnectionManager()
