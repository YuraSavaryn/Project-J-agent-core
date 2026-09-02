"""Chat endpoint — stub implementation (echo reply)."""

from fastapi import APIRouter

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
        "> **Note**: This is currently a stub that echoes the input. "
        "Full LLM integration will be added in step 6."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Echo stub — returns the user message as the agent reply.

    Will be replaced by the full orchestrator pipeline in step 6.
    """
    return ChatResponse(
        session_id=request.session_id,
        reply=f"[ECHO] {request.message}",
        sources=[],
        suggested_actions=[],
    )

