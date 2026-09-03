"""Chat endpoint — full LLM pipeline via Orchestrator."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from api.dependencies import OrchestratorDep
from api.schemas.chat import ChatRequest, ChatResponse, SourceChunk, SuggestedAction
from core.llm.router import LLMUnavailableError

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
        "The agent maintains session context across multiple calls using the same `session_id`."
    ),
)
async def chat(
    request: ChatRequest,
    orchestrator: OrchestratorDep,
) -> ChatResponse:
    """Handle a user message through the full LLM orchestrator pipeline.

    Pipeline: session history → prompt builder → LLM router → session persist → response.
    """
    try:
        result = await orchestrator.handle_message(
            session_id=request.session_id,
            user_message=request.message,
        )
    except LLMUnavailableError:
        logger.error(
            "All LLM providers unavailable for session=%s", request.session_id
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Наразі сервіс тимчасово недоступний. "
                "Спробуйте ще раз через кілька хвилин."
            ),
        )

    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        sources=[
            SourceChunk(**s) for s in result.sources
        ],
        suggested_actions=[
            SuggestedAction(**a) for a in result.suggested_actions
        ],
    )
