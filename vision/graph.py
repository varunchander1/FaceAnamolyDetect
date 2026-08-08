"""LangGraph orchestration: multimodal VLM node + empty-answer fallback loop.

Graph shape:  START -> vlm -> (check) -> END | fallback -> vlm

State carries a persisted `messages` history (multi-turn memory via the
checkpointer) plus the transient per-question inputs. Stored history keeps only
text (question + answer) — images are sent for the current turn only, so context
never balloons across a session.
"""

from __future__ import annotations

from typing import Annotated, Iterator, TypedDict, Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

from vision.answer import tidy
from vision.llm import build_human_message, build_system_message, make_chat
from vision.settings import settings

# Models tried, in order, when the primary returns an empty answer.
FALLBACK_CHAIN = settings.fallback_chain


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # persisted chat history (memory)
    frames: list  # current turn's BGR frames (transient)
    context: str  # current scene-grounding text (transient)
    question: str
    model: str  # model to use now; fallback mutates it
    answer: str  # last model output
    attempts: int  # fallback counter
    fallback_chain: list[str]  # per-call override of FALLBACK_CHAIN


def _chain(state: State) -> list[str]:
    """Per-call fallback chain, defaulting to the env-configured one."""
    return list(state.get("fallback_chain") or FALLBACK_CHAIN)


def _vlm(state: State) -> dict:
    """
    Vision Language Model node.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary containing the updated state.

    Raises:
        ValueError: If the model is not a valid model.

    Example:
        >>> state = {
        ...     "model": "gpt-5-mini",
        ...     "frames": [...],
        ...     "context": "...",
        ...     "question": "..."}
        >>> _vlm(state)
        {'answer': '...', 'messages': [HumanMessage(content='...'), AIMessage(content='...')]}
    """
    model = state["model"]
    system = build_system_message(model)
    human = build_human_message(
        state["frames"], state["context"], state["question"], model
    )
    prompt = [system] + list(state["messages"]) + [human]
    ai = make_chat(model).invoke(prompt)
    content = ai.content if isinstance(ai.content, str) else str(ai.content)
    # tidy() is the backstop for models that ignore the brevity/no-preamble
    # rules in INSTRUCTIONS — the event feed shows short answers only.
    answer = tidy(content)

    update: dict = {"answer": answer}
    if answer:  # keep failed retries out of history; store text only
        update["messages"] = [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ]
    return update


def _check(state: State) -> str:
    if state["answer"].strip():
        return END
    if state["attempts"] < len(_chain(state)):
        return "fallback"
    return END


def _fallback(state: State) -> dict:
    i = state["attempts"]
    return {"model": _chain(state)[i], "attempts": i + 1}


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[State, None, State, State]:
    """
    Build the graph.

    Args:
        checkpointer: Optional saver enabling multi-turn memory. Pass
            `InMemorySaver()` for a conversational caller (invocations then need
            a `thread_id`). Leave `None` for stateless one-shot invocations —
            `InMemorySaver` never evicts, so a caller minting a fresh thread_id
            per call would accumulate checkpoints forever.

    Returns:
        A CompiledStateGraph object.
    """
    g = StateGraph(state_schema=State)
    g.add_node("vlm", _vlm)
    g.add_node("fallback", _fallback)
    g.add_edge(START, "vlm")
    g.add_conditional_edges("vlm", _check, {"fallback": "fallback", END: END})
    g.add_edge("fallback", "vlm")
    return g.compile(checkpointer=checkpointer)


def stream_answer_tokens(
    graph, initial_state: dict, thread_id: str
) -> Iterator[str | list[str | dict[Any, Any]]]:
    """
    Yield answer tokens as they stream. Requires a graph built with a checkpointer.

    Unused by the Streamlit app (probes are one-shot `invoke` calls); kept for
    conversational callers that want token-by-token output.

    Args:
        graph: The graph to stream from.
        initial_state: The initial state of the graph. Must contain "messages" and "answer" keys.
        thread_id: The thread ID to use for the stream.

    Yields:
        str: Answer tokens as they stream.
        list[str | dict[Any, Any]]: Additional metadata about the streamed tokens.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    for chunk, _meta in graph.stream(initial_state, cfg, stream_mode="messages"):
        # Only live LLM token partials — skip the full HumanMessage/AIMessage
        # we append to state (those would duplicate the streamed text).
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            yield chunk.content
