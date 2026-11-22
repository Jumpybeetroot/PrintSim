"""ToS-compliant official Discord Bot channel scanner and review-queue exporter.

Security & Terms-of-Service Policy:
- This module ONLY supports official Discord Bot tokens passed via ``DISCORD_BOT_TOKEN``
  or ``--bot-token`` and always authenticates with ``Authorization: Bot <token>``.
- User-account tokens ("self-botting") are explicitly unsupported and prohibited.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from printsim.kb.extract import (
    _SHAPER_ATTACH_EXT_RE,
    _SHAPER_ATTACH_NAME_RE,
    score_engineering_signal,
)
from printsim.kb.store import TouchstoneKB

DISCORD_API_BASE = "https://discord.com/api/v10"
DISCORD_USER_AGENT = "DiscordBot (https://github.com/printsim, 1.0)"


def resolve_bot_token(bot_token: str | None = None) -> str:
    """Resolve and validate an official Discord Bot token (never a user self-bot token)."""
    raw = (bot_token if bot_token is not None else os.environ.get("DISCORD_BOT_TOKEN", "")).strip()
    if not raw:
        raise ValueError(
            "Missing Discord Bot token. Set DISCORD_BOT_TOKEN or pass --bot-token. "
            "Only official Discord Bot tokens are supported; user-account tokens (self-botting) are prohibited."
        )
    # Strip optional leading "Bot " prefix cleanly; reject Bearer/user-token markers or whitespace.
    if raw.lower().startswith("bot "):
        raw = raw[4:].strip()
    if raw.lower().startswith("bearer ") or raw.lower().startswith("mfa."):
        raise ValueError(
            "Rejected non-bot token format. Only official Discord Bot tokens are supported; "
            "user-account tokens (self-botting) are prohibited."
        )
    if any(ch.isspace() for ch in raw):
        raise ValueError("Invalid Discord Bot token: contains whitespace.")
    return raw


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "attachment").strip("._")
    return safe or "attachment"


def _is_shaper_attachment(filename: str) -> bool:
    return bool(_SHAPER_ATTACH_EXT_RE.search(filename) and _SHAPER_ATTACH_NAME_RE.search(filename))


def _fetch_discord_page(
    url: str,
    headers: dict[str, str],
    sleep_fn: Callable[[float], None],
    max_retries: int = 5,
    timeout_s: float = 20.0,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch one paginated batch of Discord messages with full HTTP 429 & rate-limit header handling."""
    rate_waits = 0
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                messages = json.loads(body)
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset_after = resp.headers.get("X-RateLimit-Reset-After")
                if remaining is not None and reset_after is not None:
                    try:
                        if int(remaining) <= 0:
                            wait_s = max(0.0, float(reset_after))
                            if wait_s > 0:
                                rate_waits += 1
                                sleep_fn(wait_s)
                    except ValueError:
                        pass
                return (messages if isinstance(messages, list) else [], rate_waits)
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < max_retries:
                retry_after = 1.0
                try:
                    err_body = err.read().decode("utf-8", errors="replace")
                    parsed = json.loads(err_body)
                    if "retry_after" in parsed:
                        retry_after = float(parsed["retry_after"])
                except Exception:  # noqa: BLE001
                    hdr_retry = err.headers.get("Retry-After") if err.headers else None
                    if hdr_retry:
                        try:
                            retry_after = float(hdr_retry)
                        except ValueError:
                            pass
                rate_waits += 1
                sleep_fn(max(0.0, retry_after))
                continue
            raise
    return ([], rate_waits)


def _download_attachment(
    url: str,
    dest_path: Path,
    timeout_s: float = 20.0,
) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": DISCORD_USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(resp.read())
        return True
    except Exception:  # noqa: BLE001
        return False


def scan_discord_channel(
    kb: TouchstoneKB,
    channel_id: str,
    bot_token: str | None = None,
    limit: int = 500,
    min_score: float = 0.25,
    download_attachments_dir: Path | None = None,
    review_out_dir: Path | None = None,
    api_base: str = DISCORD_API_BASE,
    sleep_fn: Callable[[float], None] = time.sleep,
    page_size: int = 100,
) -> dict[str, Any]:
    """Scan a Discord channel using an official Bot token, filter high-signal posts, and export review queue.

    Parameters
    ----------
    kb:
        Target ``TouchstoneKB`` database.
    channel_id:
        Numeric Discord channel ID the authorized bot can read.
    bot_token:
        Official Discord Bot token (falls back to ``DISCORD_BOT_TOKEN`` env var).
        Always sent as ``Authorization: Bot <token>``.
    limit:
        Maximum number of recent messages to paginate through.
    min_score:
        Minimum engineering signal score in ``[0, 1]`` to store as ``pending``.
    download_attachments_dir:
        Optional directory where matching shaper/resonance PNG/CSV attachments are saved.
    review_out_dir:
        Optional directory where ``review_queue.json`` and ``review_queue.md`` are exported.
    """
    token = resolve_bot_token(bot_token)
    chan_id = str(channel_id).strip()
    if not chan_id:
        raise ValueError("channel_id must not be empty.")

    headers = {
        "Authorization": f"Bot {token}",
        "User-Agent": DISCORD_USER_AGENT,
        "Accept": "application/json",
    }

    messages_scanned = 0
    pages_fetched = 0
    rate_limit_waits = 0
    items_kept = 0
    items_deduplicated = 0
    attachments_downloaded = 0

    before_id: str | None = None
    remaining_target = max(1, int(limit))
    max_page = max(1, min(100, int(page_size)))

    while remaining_target > 0:
        batch_size = min(max_page, remaining_target)
        url = f"{api_base.rstrip('/')}/channels/{chan_id}/messages?limit={batch_size}"
        if before_id is not None:
            url += f"&before={before_id}"

        batch, waits = _fetch_discord_page(url, headers=headers, sleep_fn=sleep_fn)
        pages_fetched += 1
        rate_limit_waits += waits

        if not batch:
            break

        for msg in batch:
            messages_scanned += 1
            msg_id = str(msg.get("id") or "")
            content = str(msg.get("content") or "")
            author_obj = msg.get("author") or {}
            author_name = (
                str(author_obj.get("global_name") or author_obj.get("username") or "")
                if isinstance(author_obj, dict)
                else str(author_obj)
            )
            timestamp = str(msg.get("timestamp") or "")
            raw_atts = msg.get("attachments") or []
            norm_atts: list[dict[str, Any]] = []
            for att in raw_atts:
                if not isinstance(att, dict):
                    continue
                norm_atts.append(
                    {
                        "filename": str(att.get("filename") or ""),
                        "url": str(att.get("url") or ""),
                        "content_type": str(att.get("content_type") or ""),
                        "size": att.get("size"),
                    }
                )

            score, tags = score_engineering_signal(content, attachments=norm_atts)
            if score < min_score:
                continue

            if download_attachments_dir is not None:
                dl_dir = Path(download_attachments_dir)
                for att in norm_atts:
                    fname = att["filename"]
                    att_url = att["url"]
                    if att_url and _is_shaper_attachment(fname):
                        local_name = f"{msg_id}_{_sanitize_filename(fname)}" if msg_id else _sanitize_filename(fname)
                        local_dest = dl_dir / local_name
                        if _download_attachment(att_url, local_dest):
                            att["local_path"] = str(local_dest)
                            attachments_downloaded += 1

            guild_id = msg.get("guild_id")
            msg_url = (
                f"https://discord.com/channels/{guild_id}/{chan_id}/{msg_id}"
                if guild_id and msg_id
                else f"https://discord.com/channels/@me/{chan_id}/{msg_id}" if msg_id else ""
            )

            _, inserted = kb.upsert_raw_item(
                source_type="discord_bot",
                source_name="discord",
                channel=chan_id,
                author=author_name,
                url=msg_url,
                published_at=timestamp,
                title=f"Discord #{chan_id} — {author_name or 'user'} ({msg_id})",
                raw_text=content,
                attachments=norm_atts,
                signal_score=score,
                signal_tags=tags,
                review_status="pending",
            )
            if inserted:
                items_kept += 1
            else:
                items_deduplicated += 1

        remaining_target -= len(batch)
        last_id = batch[-1].get("id")
        if len(batch) < batch_size or not last_id:
            break
        before_id = str(last_id)

    review_export: dict[str, Any] | None = None
    if review_out_dir is not None:
        review_export = kb.export_review_queue(Path(review_out_dir), min_score=min_score)

    return {
        "channel_id": chan_id,
        "messages_scanned": messages_scanned,
        "pages_fetched": pages_fetched,
        "rate_limit_waits": rate_limit_waits,
        "items_kept": items_kept,
        "items_deduplicated": items_deduplicated,
        "attachments_downloaded": attachments_downloaded,
        "review_export": review_export,
    }
