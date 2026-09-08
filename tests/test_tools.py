"""Tests for tool calling and registry."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.llm.base import LLMResponse
from core.llm.router import LLMRouter
from core.memory.session_store import SessionStore
from core.orchestrator import Orchestrator
from tools.base import Tool
from tools.registry import ToolRegistry


class DummyWeatherTool(Tool):
    name = "get_weather"
    description = "Get the current weather."
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}}

    async def run(self, city: str, **kwargs) -> str:
        if city == "Kyiv":
            return "Сонячно, 25 градусів."
        return "Невідома погода."


@pytest.mark.asyncio
async def test_tool_registry():
    registry = ToolRegistry()
    registry.register(DummyWeatherTool())
    
    # Check registration
    assert len(registry) == 1
    assert "get_weather" in registry
    assert registry.get("get_weather") is not None
    
    # Check schemas
    schemas = registry.to_openai_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "get_weather"
    
    # Check replace
    with pytest.raises(ValueError, match="already registered"):
        registry.register(DummyWeatherTool())
    registry.replace(DummyWeatherTool())


@pytest.mark.asyncio
async def test_orchestrator_tool_loop():
    # Mock store
    store = AsyncMock(spec=SessionStore)
    store.get_history.return_value = []
    
    # Tool registry
    registry = ToolRegistry()
    registry.register(DummyWeatherTool())
    
    # Mock LLM router
    llm_router = AsyncMock(spec=LLMRouter)
    
    # First response: tool_call asking for weather in Kyiv
    res1 = LLMResponse(
        content=None,
        provider="mock",
        model="mock",
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Kyiv"})
                }
            }
        ]
    )
    
    # Second response: final answer after tool
    res2 = LLMResponse(
        content="Погода в Києві: сонячно, 25 градусів.",
        provider="mock",
        model="mock",
        tool_calls=[]
    )
    
    # Set side_effect so it returns res1 then res2
    llm_router.complete.side_effect = [res1, res2]
    
    orchestrator = Orchestrator(
        session_store=store,
        llm_router=llm_router,
        tool_registry=registry,
    )
    
    result = await orchestrator.handle_message(
        session_id="test-session",
        user_message="Яка погода в Києві?"
    )
    
    # Check final result
    assert result.reply == "Погода в Києві: сонячно, 25 градусів."
    
    # Check that complete was called twice
    assert llm_router.complete.call_count == 2
    
    # Check the second call to LLM: it should have the tool result
    second_call_messages = llm_router.complete.call_args_list[1][0][0]
    
    # System(1), User(1), Assistant Tool Call(1), Tool Result(1)
    assert len(second_call_messages) >= 4
    assert second_call_messages[-2]["role"] == "assistant"
    assert "tool_calls" in second_call_messages[-2]
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["content"] == "Сонячно, 25 градусів."
