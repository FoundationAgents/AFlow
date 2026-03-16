"""Capture real Eve responses for a dataset split.

Sends each user_message from eve_{split}.jsonl to Eve's WebSocket chat endpoint,
captures the response, and writes an updated JSONL with eve_baseline_response values.

Usage:
    uv run python scripts/capture_eve_baseline.py --split dev [--host localhost] [--port 8080]

Requires Eve to be running (docker-compose up).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import requests
import websockets

DATA_DIR = Path("data/datasets")


def get_token(base_url: str, email: str) -> str:
    """Get JWT token from Eve's token endpoint."""
    resp = requests.post(f"{base_url}/chat/token", json={"email": email})
    resp.raise_for_status()
    return resp.json()["token"]


async def send_and_receive(
    ws_url: str, token: str, message: str, timeout: float = 60.0
) -> str:
    """Open a fresh WebSocket connection, send a message, and wait for Eve's response."""
    uri = f"{ws_url}?timezone=UTC"
    subprotocols = ["atlas-web-chat.v3", f"jwt.{token}"]

    async with websockets.connect(uri, subprotocols=subprotocols) as ws:
        # Send user message
        await ws.send(json.dumps({"__chat_event": "user-message", "content": message}))

        # Wait for aicw-message response
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
                data = json.loads(raw)
                if data.get("__chat_event") == "aicw-message":
                    return data["content"]
            except asyncio.TimeoutError:
                break

    return "[NO RESPONSE — TIMEOUT]"


async def capture_one(
    idx: int,
    total: int,
    sample: dict,
    base_url: str,
    ws_url: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Capture Eve's response for a single sample (semaphore-bounded)."""
    async with semaphore:
        user_msg = sample["user_message"]
        scenario = sample.get("scenario", "general")
        conv_id = sample.get("conversation_id", f"conv_{idx}")

        print(f"[{idx+1}/{total}] ({scenario}) {user_msg[:80]}...")

        context = sample.get("context", "")
        try:
            if context:
                token = get_token(base_url, f"{conv_id}@baseline.test")
                response = await _replay_multiturn(ws_url, token, context, user_msg)
            else:
                token = get_token(base_url, f"{conv_id}@baseline.test")
                response = await send_and_receive(ws_url, token, user_msg)
        except Exception as e:
            response = f"[ERROR: {e}]"

        print(f"  [{idx+1}] → {response[:100]}...")

        updated = dict(sample)
        updated["eve_baseline_response"] = response
        return updated


async def capture_all(
    host: str, port: int, split: str = "validate", concurrency: int = 10
):
    base_url = f"http://{host}:{port}"
    ws_url = f"ws://{host}:{port}/chat/ws"

    input_path = DATA_DIR / f"eve_{split}.jsonl"
    output_path = DATA_DIR / f"eve_{split}_with_baseline.jsonl"

    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))

    total = len(samples)
    print(f"Loaded {total} samples from {input_path}")
    print(f"Eve endpoint: {base_url}")
    print(f"Concurrency: {concurrency}")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        capture_one(i, total, sample, base_url, ws_url, semaphore)
        for i, sample in enumerate(samples)
    ]
    results = await asyncio.gather(*tasks)

    # Write output (preserve original order)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(results)} samples to {output_path}")


async def _replay_multiturn(
    ws_url: str, token: str, context: str, final_message: str
) -> str:
    """Replay a multi-turn conversation: send prior user turns, then the final message."""
    uri = f"{ws_url}?timezone=UTC"
    subprotocols = ["atlas-web-chat.v3", f"jwt.{token}"]

    async with websockets.connect(uri, subprotocols=subprotocols) as ws:
        # Parse and replay prior user turns from context
        prior_user_msgs = []
        for line in context.split("\n"):
            line = line.strip()
            if line.startswith("User:"):
                prior_user_msgs.append(line[len("User:") :].strip())

        # Send each prior user message and wait for Eve's response
        for msg in prior_user_msgs:
            await ws.send(json.dumps({"__chat_event": "user-message", "content": msg}))
            # Wait for Eve to finish responding
            deadline = time.time() + 60.0
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=deadline - time.time()
                    )
                    data = json.loads(raw)
                    if data.get("__chat_event") == "aicw-message":
                        break  # Got response, move to next turn
                except asyncio.TimeoutError:
                    break
            await asyncio.sleep(1)

        # Now send the final user message and capture the response
        await ws.send(
            json.dumps({"__chat_event": "user-message", "content": final_message})
        )
        deadline = time.time() + 60.0
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
                data = json.loads(raw)
                if data.get("__chat_event") == "aicw-message":
                    return data["content"]
            except asyncio.TimeoutError:
                break

    return "[NO RESPONSE — TIMEOUT]"


def main():
    parser = argparse.ArgumentParser(description="Capture Eve baseline responses")
    parser.add_argument(
        "--split",
        default="validate",
        help="Dataset split to capture (e.g. validate, dev, test, train)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max parallel WebSocket connections",
    )
    args = parser.parse_args()

    asyncio.run(capture_all(args.host, args.port, args.split, args.concurrency))


if __name__ == "__main__":
    main()
