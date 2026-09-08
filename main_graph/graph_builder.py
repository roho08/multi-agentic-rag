"""Main entrypoint for the conversational retrieval graph."""

from typing import Any, Literal, TypedDict, Optional

import logging

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from main_graph.graph_states import (
    AgentState,
    Router,
    GradeHallucinations,
    InputState,
)

from utils.prompt import (
    ROUTER_SYSTEM_PROMPT,
    RESEARCH_PLAN_SYSTEM_PROMPT,
    MORE_INFO_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    RESPONSE_SYSTEM_PROMPT,
)

from subgraph.graph_builder import researcher_graph
from utils.local_llm import chat_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Helper: invoke local LLM
# ============================================================

def invoke_model(messages):
    """
    Invoke the local Hugging Face chat model synchronously.

    Local Hugging Face models do not need the OpenAI-style
    async_client / structured-output machinery.
    """
    return chat_model.invoke(messages)


def extract_text(response) -> str:
    """Extract plain text from a LangChain AIMessage."""
    if hasattr(response, "content"):
        return response.content

    return str(response)


# ============================================================
# Router
# ============================================================

def analyze_and_route_query(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:

    messages = [
        {
            "role": "system",
            "content": ROUTER_SYSTEM_PROMPT,
        }
    ] + state.messages

    logger.info("--- ANALYZE AND ROUTE QUERY ---")
    logger.info("MESSAGES: %s", state.messages)

    response = invoke_model(messages)
    text = extract_text(response)

    logger.info("ROUTER RAW RESPONSE: %s", text)

    # Temporary simple parsing.
    # We will improve this after confirming the LLM output.
    text_lower = text.lower()

    if "more-info" in text_lower:
        router_type = "more-info"
    elif "environmental" in text_lower:
        router_type = "environmental"
    else:
        router_type = "general"

    router = {
        "type": router_type,
        "logic": text,
    }

    return {"router": router}


def route_query(
    state: AgentState,
) -> Literal[
    "create_research_plan",
    "ask_for_more_info",
    "respond_to_general_query",
]:

    router_type = state.router["type"]

    if router_type == "environmental":
        return "create_research_plan"

    elif router_type == "more-info":
        return "ask_for_more_info"

    elif router_type == "general":
        return "respond_to_general_query"

    else:
        raise ValueError(f"Unknown router type: {router_type}")


# ============================================================
# Research Plan
# ============================================================

class Plan(TypedDict):
    steps: list[str]


def create_research_plan(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:

    messages = [
        {
            "role": "system",
            "content": RESEARCH_PLAN_SYSTEM_PROMPT,
        }
    ] + state.messages

    logger.info("--- PLAN GENERATION ---")

    response = invoke_model(messages)
    text = extract_text(response)

    logger.info("PLAN RAW RESPONSE: %s", text)

    # Temporary parsing.
    # We will make the model return clean JSON in the next step.
    lines = [
        line.strip("- •123456789. ")
        for line in text.splitlines()
        if line.strip()
    ]

    steps = lines[:2]

    if not steps:
        steps = [state.messages[-1].content]

    return {
        "steps": steps,
        "documents": [],
    }


# ============================================================
# Ask for more information
# ============================================================

def ask_for_more_info(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage]]:

    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + state.messages

    response = invoke_model(messages)

    return {
        "messages": [response]
    }


# ============================================================
# Conduct Research
# ============================================================

async def conduct_research(
    state: AgentState,
) -> dict[str, Any]:

    result = await researcher_graph.ainvoke(
        {
            "question": state.steps[0]
        }
    )

    docs = result["documents"]
    step = state.steps[0]

    logger.info(
        "%s documents retrieved for step: %s",
        len(docs),
        step,
    )

    return {
        "documents": docs,
        "steps": state.steps[1:],
    }


def check_finished(
    state: AgentState,
) -> Literal["respond", "conduct_research"]:

    if len(state.steps or []) > 0:
        return "conduct_research"

    return "respond"


# ============================================================
# General Query
# ============================================================

def respond_to_general_query(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage]]:

    system_prompt = GENERAL_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )

    logger.info("--- GENERAL QUERY RESPONSE ---")

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + state.messages

    response = invoke_model(messages)

    return {
        "messages": [response]
    }


# ============================================================
# Document Formatting
# ============================================================

def _format_doc(doc: Document) -> str:

    metadata = doc.metadata or {}

    meta = "".join(
        f" {k}={v!r}"
        for k, v in metadata.items()
    )

    if meta:
        meta = f" {meta}"

    return (
        f"<document{meta}>\n"
        f"{doc.page_content}\n"
        f"</document>"
    )


def format_docs(
    docs: Optional[list[Document]]
) -> str:

    if not docs:
        return "<documents></documents>"

    formatted = "\n".join(
        _format_doc(doc)
        for doc in docs
    )

    return f"""<documents>
{formatted}
</documents>"""


# ============================================================
# Hallucination Check
# ============================================================

def check_hallucinations(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:

    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=format_docs(state.documents),
        generation=state.messages[-1],
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + state.messages

    logger.info("--- CHECK HALLUCINATIONS ---")

    response = invoke_model(messages)
    text = extract_text(response)

    logger.info(
        "HALLUCINATION GRADER RAW RESPONSE: %s",
        text,
    )

    # Temporary binary parsing.
    if "0" in text and "1" not in text:
        score = "0"
    else:
        score = "1"

    return {
        "hallucination": {
            "binary_score": score
        }
    }


# ============================================================
# Human Approval
# ============================================================

def human_approval(
    state: AgentState,
):

    binary_score = state.hallucination["binary_score"]

    if binary_score == "1":
        return "END"

    retry_generation = interrupt(
        {
            "question": "Is this correct?",
            "llm_output": state.messages[-1],
        }
    )

    if retry_generation == "y":
        return "respond"

    return "END"


# ============================================================
# Final Response
# ============================================================

def respond(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage]]:

    logger.info("--- RESPONSE GENERATION STEP ---")

    context = format_docs(state.documents)

    prompt = RESPONSE_SYSTEM_PROMPT.format(
        context=context
    )

    messages = [
        {
            "role": "system",
            "content": prompt,
        }
    ] + state.messages

    response = invoke_model(messages)

    return {
        "messages": [response]
    }


# ============================================================
# LangGraph
# ============================================================

checkpointer = MemorySaver()

builder = StateGraph(
    AgentState,
    input=InputState,
)

builder.add_node(
    "analyze_and_route_query",
    analyze_and_route_query,
)

builder.add_edge(
    START,
    "analyze_and_route_query",
)

builder.add_conditional_edges(
    "analyze_and_route_query",
    route_query,
)

builder.add_node(
    "create_research_plan",
    create_research_plan,
)

builder.add_node(
    "ask_for_more_info",
    ask_for_more_info,
)

builder.add_node(
    "respond_to_general_query",
    respond_to_general_query,
)

builder.add_node(
    "conduct_research",
    conduct_research,
)

builder.add_node(
    "respond",
    respond,
)

builder.add_node(
    "check_hallucinations",
    check_hallucinations,
)

builder.add_conditional_edges(
    "check_hallucinations",
    human_approval,
    {
        "END": END,
        "respond": "respond",
    },
)

builder.add_edge(
    "create_research_plan",
    "conduct_research",
)

builder.add_conditional_edges(
    "conduct_research",
    check_finished,
)

graph = builder.compile(
    checkpointer=checkpointer
)