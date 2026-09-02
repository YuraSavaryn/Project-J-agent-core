"""Chat endpoint — stub implementation that echoes and stores history."""

from fastapi import APIRouter

from api.dependencies import SessionStoreDep
from api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the agent",
    description=(
        "Main conversational endpoint. "
        "Accepts a user message and returns the agent's reply with optional sources "
        "and suggested follow-up actions.\n\n"
        "> **Note**: This is an echo stub with session history tracking. "
        "Full LLM integration will be added in step 6."
    ),
)
async def chat(
    request: ChatRequest,
    session_store: SessionStoreDep,
) -> ChatResponse:
    """Echo stub — stores user+assistant messages and returns echo reply.

    Will be replaced by the full orchestrator pipeline in step 6.
    """
    # 1. Persist user message to session history
    await session_store.append_message(
        session_id=request.session_id,
        role="user",
        content=request.message,
    )

    reply = f"[ECHO] {request.message}"

    # 2. Persist assistant reply to session history
    await session_store.append_message(
        session_id=request.session_id,
        role="assistant",
        content=reply,
    )

    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        sources=[],
        suggested_actions=[],
    )
