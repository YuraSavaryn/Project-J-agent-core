"""Chat endpoint — full LLM pipeline via Orchestrator."""

import logging

from fastapi import APIRouter, Header

from api.dependencies import OrchestratorDep
from api.schemas.chat import ChatRequest, ChatResponse, SourceChunk, SuggestedAction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the agent",
    description=(
        "Main conversational endpoint. "
        "Accepts a user message and returns the agent's reply with optional sources "
        "and suggested follow-up actions.\n\n"
        "The agent maintains session context across multiple calls using the same `session_id`.\n\n"
        "Optionally supply `X-User-Id` header to identify the external user "
        "(Telegram chat_id, web cookie, …).\n\n"
        "On LLM provider failure returns **503** with `{error: {code: 'llm_unavailable', ...}}`."
    ),
)
async def chat(
    request: ChatRequest,
    orchestrator: OrchestratorDep,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> ChatResponse:
    """Handle a user message through the full LLM orchestrator pipeline.

    Pipeline: session history → prompt builder → LLM router → session persist → response.

    Raises ``LLMUnavailableError`` when all providers fail; the global exception
    handler in ``api/main.py`` converts it to a 503 ``ErrorResponse``.
    """
    external_user_id = x_user_id or "anonymous"

    result = await orchestrator.handle_message(
        session_id=request.session_id,
        user_message=request.message,
        channel=request.channel,
        external_user_id=external_user_id,
    )

    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        sources=[SourceChunk(**s) for s in result.sources],
        suggested_actions=[SuggestedAction(**a) for a in result.suggested_actions],
        query_log_id=result.query_log_id,
    )
