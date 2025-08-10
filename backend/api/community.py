#!/usr/bin/env python3
"""
Community API: wallet-gated posts, likes, comments, and replies.

Storage: PostgreSQL via existing async pool in backend.accounting.db_config.
Auth: Accepts Blyan wallet session tokens (from /wallet/authenticate). Also
      supports SIWE tokens if present by validating against Redis session.

Schema (created on startup if missing, under current search_path schema):
  - community_posts(id BIGSERIAL PK, author_address TEXT, title TEXT, content TEXT,
                    like_count INT, comment_count INT, created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ)
  - community_comments(id BIGSERIAL PK, post_id BIGINT FK, author_address TEXT,
                       content TEXT, parent_comment_id BIGINT NULL,
                       created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)
  - community_likes(post_id BIGINT, user_address TEXT, created_at TIMESTAMPTZ,
                    PRIMARY KEY(post_id, user_address))
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.accounting.db_config import db, init_database


router = APIRouter(prefix="/community", tags=["community"])


# ---------------------------- Models ----------------------------


class CreatePostRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=10_000)


class PostSummary(BaseModel):
    id: int
    author_address: str
    title: str
    content: str
    like_count: int
    comment_count: int
    created_at: str
    updated_at: str
    liked_by_me: Optional[bool] = False


class CreateCommentRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)
    parent_comment_id: Optional[int] = None


class Comment(BaseModel):
    id: int
    post_id: int
    author_address: str
    content: str
    parent_comment_id: Optional[int]
    created_at: str
    updated_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------- Auth helpers ----------------------------


async def _validate_token_and_get_address(token: str) -> str:
    """Validate session token against wallet_auth or SIWE and return address.

    Order:
      1) wallet_auth.verify_token (in-memory session)
      2) siwe_auth.get_session (Redis)
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    # 1) Try wallet_auth
    try:
        from backend.api.wallet_auth import verify_token as wallet_verify

        result = await wallet_verify(token)  # type: ignore[func-returns-value]
        if isinstance(result, dict) and result.get("valid"):
            return str(result.get("address"))
    except Exception:
        # Fall through to SIWE
        pass

    # 2) Try SIWE
    try:
        from backend.api.siwe_auth import get_session as siwe_get_session

        session = await siwe_get_session(token)  # type: ignore[func-returns-value]
        if isinstance(session, dict) and session.get("valid"):
            return str(session.get("address"))
    except Exception:
        pass

    raise HTTPException(status_code=401, detail="Invalid or expired session token")


def _extract_bearer_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    # Fallback header
    alt = request.headers.get("X-Blyan-Token") or request.headers.get("x-blyan-token")
    if alt:
        return alt.strip()
    # Fallback query param
    token = request.query_params.get("token")
    return token.strip() if token else None


async def require_wallet_address(request: Request) -> str:
    token = _extract_bearer_token(request)
    return await _validate_token_and_get_address(token or "")


# ---------------------------- Schema setup ----------------------------


@router.on_event("startup")
async def ensure_tables():
    # Ensure DB and tables exist. Tables will be created under the configured schema
    # (search_path is set to include the configured schema in db_config).
    await init_database()
    # Create tables if not exist
    create_sql = [
        # Posts
        """
        CREATE TABLE IF NOT EXISTS community_posts (
            id BIGSERIAL PRIMARY KEY,
            author_address TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            like_count INT NOT NULL DEFAULT 0,
            comment_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_community_posts_created_at ON community_posts (created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_community_posts_author ON community_posts (author_address);
        """,
        # Comments
        """
        CREATE TABLE IF NOT EXISTS community_comments (
            id BIGSERIAL PRIMARY KEY,
            post_id BIGINT NOT NULL REFERENCES community_posts(id) ON DELETE CASCADE,
            author_address TEXT NOT NULL,
            content TEXT NOT NULL,
            parent_comment_id BIGINT NULL REFERENCES community_comments(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_community_comments_post ON community_comments (post_id);
        CREATE INDEX IF NOT EXISTS idx_community_comments_parent ON community_comments (parent_comment_id);
        """,
        # Likes
        """
        CREATE TABLE IF NOT EXISTS community_likes (
            post_id BIGINT NOT NULL REFERENCES community_posts(id) ON DELETE CASCADE,
            user_address TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (post_id, user_address)
        );
        CREATE INDEX IF NOT EXISTS idx_community_likes_user ON community_likes (user_address);
        """,
    ]

    for sql in create_sql:
        await db.execute(sql)


# ---------------------------- Endpoints: Posts ----------------------------


@router.get("/posts", response_model=List[PostSummary])
async def list_posts(
    request: Request,
    limit: int = 20,
    offset: int = 0,
):
    limit = max(1, min(50, limit))
    offset = max(0, offset)

    rows = await db.fetchall(
        """
        SELECT id, author_address, title, content, like_count, comment_count,
               created_at, updated_at
        FROM community_posts
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit,
        offset,
    )

    # Optional liked_by_me flag
    liked_map: Dict[int, bool] = {}
    token = _extract_bearer_token(request)
    user_address: Optional[str] = None
    if token:
        try:
            user_address = await _validate_token_and_get_address(token)
        except Exception:
            user_address = None

    if user_address and rows:
        post_ids = [r["id"] for r in rows]
        placeholders = ",".join([f"${i}" for i in range(1, len(post_ids) + 2)])
        params: List[Any] = [*post_ids, user_address]
        liked_rows = await db.fetchall(
            f"""
            SELECT post_id FROM community_likes
            WHERE post_id IN ({",".join([f"${i}" for i in range(1, len(post_ids)+1)])})
              AND user_address = ${len(post_ids)+1}
            """,
            *params,
        )
        liked_map = {lr["post_id"]: True for lr in liked_rows}

    result: List[PostSummary] = []
    for r in rows:
        result.append(
            PostSummary(
                id=r["id"],
                author_address=r["author_address"],
                title=r["title"],
                content=r["content"],
                like_count=int(r["like_count"]),
                comment_count=int(r["comment_count"]),
                created_at=r["created_at"].isoformat() if r["created_at"] else _now_iso(),
                updated_at=r["updated_at"].isoformat() if r["updated_at"] else _now_iso(),
                liked_by_me=liked_map.get(r["id"], False),
            )
        )
    return result


@router.post("/posts", response_model=PostSummary, status_code=status.HTTP_201_CREATED)
async def create_post(payload: CreatePostRequest, address: str = Depends(require_wallet_address)):
    # Basic content safety: trim and collapse excessive whitespace
    title = re.sub(r"\s+", " ", payload.title).strip()
    content = payload.content.strip()

    async with db.transaction() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO community_posts (author_address, title, content)
            VALUES ($1, $2, $3)
            RETURNING id, author_address, title, content, like_count, comment_count, created_at, updated_at
            """,
            address.lower(),
            title,
            content,
        )

    return PostSummary(
        id=row["id"],
        author_address=row["author_address"],
        title=row["title"],
        content=row["content"],
        like_count=int(row["like_count"]),
        comment_count=int(row["comment_count"]),
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
        liked_by_me=False,
    )


@router.get("/posts/{post_id}", response_model=PostSummary)
async def get_post(post_id: int, request: Request):
    row = await db.fetchone(
        """
        SELECT id, author_address, title, content, like_count, comment_count,
               created_at, updated_at
        FROM community_posts
        WHERE id = $1
        """,
        post_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")

    liked_by_me = False
    token = _extract_bearer_token(request)
    if token:
        try:
            user_address = await _validate_token_and_get_address(token)
            liked_by_me = bool(
                await db.fetchval(
                    "SELECT 1 FROM community_likes WHERE post_id=$1 AND user_address=$2",
                    post_id,
                    user_address,
                )
            )
        except Exception:
            liked_by_me = False

    return PostSummary(
        id=row["id"],
        author_address=row["author_address"],
        title=row["title"],
        content=row["content"],
        like_count=int(row["like_count"]),
        comment_count=int(row["comment_count"]),
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
        liked_by_me=liked_by_me,
    )


# ---------------------------- Endpoints: Likes ----------------------------


class LikeResponse(BaseModel):
    liked: bool
    like_count: int


@router.post("/posts/{post_id}/like", response_model=LikeResponse)
async def toggle_like(post_id: int, address: str = Depends(require_wallet_address)):
    # Ensure post exists
    exists = await db.fetchval("SELECT 1 FROM community_posts WHERE id = $1", post_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Post not found")

    async with db.transaction() as conn:
        already = await conn.fetchval(
            "SELECT 1 FROM community_likes WHERE post_id=$1 AND user_address=$2",
            post_id,
            address.lower(),
        )
        if already:
            # Unlike
            await conn.execute(
                "DELETE FROM community_likes WHERE post_id=$1 AND user_address=$2",
                post_id,
                address.lower(),
            )
            row = await conn.fetchrow(
                """
                UPDATE community_posts
                SET like_count = GREATEST(like_count - 1, 0), updated_at = NOW()
                WHERE id = $1
                RETURNING like_count
                """,
                post_id,
            )
            return LikeResponse(liked=False, like_count=int(row["like_count"]))
        else:
            # Like
            await conn.execute(
                "INSERT INTO community_likes (post_id, user_address) VALUES ($1, $2)",
                post_id,
                address.lower(),
            )
            row = await conn.fetchrow(
                """
                UPDATE community_posts
                SET like_count = like_count + 1, updated_at = NOW()
                WHERE id = $1
                RETURNING like_count
                """,
                post_id,
            )
            return LikeResponse(liked=True, like_count=int(row["like_count"]))


# ---------------------------- Endpoints: Comments ----------------------------


@router.get("/posts/{post_id}/comments", response_model=List[Comment])
async def list_comments(post_id: int):
    rows = await db.fetchall(
        """
        SELECT id, post_id, author_address, content, parent_comment_id, created_at, updated_at
        FROM community_comments
        WHERE post_id = $1
        ORDER BY created_at ASC
        """,
        post_id,
    )
    result: List[Comment] = []
    for r in rows:
        result.append(
            Comment(
                id=r["id"],
                post_id=r["post_id"],
                author_address=r["author_address"],
                content=r["content"],
                parent_comment_id=r["parent_comment_id"],
                created_at=r["created_at"].isoformat() if r["created_at"] else _now_iso(),
                updated_at=r["updated_at"].isoformat() if r["updated_at"] else _now_iso(),
            )
        )
    return result


@router.post("/posts/{post_id}/comments", response_model=Comment, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: int,
    payload: CreateCommentRequest,
    address: str = Depends(require_wallet_address),
):
    # Validate parent comment belongs to same post if provided
    if payload.parent_comment_id:
        owner_post_id = await db.fetchval(
            "SELECT post_id FROM community_comments WHERE id = $1",
            payload.parent_comment_id,
        )
        if owner_post_id != post_id:
            raise HTTPException(status_code=400, detail="Invalid parent_comment_id for this post")

    async with db.transaction() as conn:
        # Ensure post exists
        post_exists = await conn.fetchval("SELECT 1 FROM community_posts WHERE id=$1", post_id)
        if not post_exists:
            raise HTTPException(status_code=404, detail="Post not found")

        row = await conn.fetchrow(
            """
            INSERT INTO community_comments (post_id, author_address, content, parent_comment_id)
            VALUES ($1, $2, $3, $4)
            RETURNING id, post_id, author_address, content, parent_comment_id, created_at, updated_at
            """,
            post_id,
            address.lower(),
            payload.content.strip(),
            payload.parent_comment_id,
        )

        # Update post comment_count
        await conn.execute(
            "UPDATE community_posts SET comment_count = comment_count + 1, updated_at = NOW() WHERE id = $1",
            post_id,
        )

    return Comment(
        id=row["id"],
        post_id=row["post_id"],
        author_address=row["author_address"],
        content=row["content"],
        parent_comment_id=row["parent_comment_id"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


# Include router in main app (for compatibility with server include pattern)
def include_router(app):
    app.include_router(router)

