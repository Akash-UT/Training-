from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents import (
    generate_answer,
    route_query,
    run_sql_agent,
)
from app.rag import search_document


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):
    query: str

    route: str
    reasoning: str
    employee_name: str | None

    sql_agent_result: str | None
    documents: list[dict[str, Any]]

    answer: str


# ============================================================
# ROUTER
# ============================================================

def router_node(state: AgentState):
    """
    Ask Gemini to semantically decide which information source
    is required.
    """

    decision = route_query(
        state["query"]
    )

    return {
        "route": decision.route,
        "reasoning": decision.reasoning,
        "employee_name": decision.employee_name,
    }


# ============================================================
# STRUCTURED DATABASE
# ============================================================

def structured_node(state: AgentState):
    """
    Execute the question against PostgreSQL using the
    LangChain SQL agent.
    """

    result = run_sql_agent(
        state["query"]
    )

    return {
        "sql_agent_result": result,
    }


# ============================================================
# DOCUMENT RAG
# ============================================================

def rag_node(state: AgentState):
    """
    Retrieve relevant document chunks from Qdrant.
    """

    documents = search_document(
        state["query"]
    )

    return {
        "documents": documents,
    }


# ============================================================
# FINAL ANSWER
# ============================================================

def answer_node(state: AgentState):
    """
    Generate the final answer using the results collected
    from PostgreSQL and/or Qdrant.
    """

    answer = generate_answer(
        query=state["query"],
        sql_agent_result=state.get(
            "sql_agent_result"
        ),
        documents=state.get(
            "documents",
            [],
        ),
    )

    return {
        "answer": answer,
    }


# ============================================================
# GENERAL QUESTIONS
# ============================================================

def general_node(state: AgentState):
    """
    Handle questions that do not require company data.
    """

    answer = generate_answer(
        query=state["query"],
        sql_agent_result=None,
        documents=[],
    )

    return {
        "answer": answer,
    }


# ============================================================
# ROUTING LOGIC
# ============================================================

def route_after_router(state: AgentState):
    """
    Decide which source should execute after Gemini routing.
    """

    route = state["route"]

    if route == "structured":
        return "structured"

    if route == "unstructured":
        return "rag"

    if route == "hybrid":
        return "structured"

    return "general"


def route_after_structured(state: AgentState):
    """
    For hybrid questions, retrieve document information after
    the structured database step.

    For structured-only questions, go directly to the final
    answer.
    """

    if state["route"] == "hybrid":
        return "rag"

    return "answer"


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(
    AgentState
)


# ============================================================
# NODES
# ============================================================

builder.add_node(
    "router",
    router_node,
)

builder.add_node(
    "structured",
    structured_node,
)

builder.add_node(
    "rag",
    rag_node,
)

builder.add_node(
    "answer",
    answer_node,
)

builder.add_node(
    "general",
    general_node,
)


# ============================================================
# START
# ============================================================

builder.add_edge(
    START,
    "router",
)


# ============================================================
# ROUTER → SOURCE
# ============================================================

builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "structured": "structured",
        "rag": "rag",
        "general": "general",
    },
)


# ============================================================
# STRUCTURED → RAG OR ANSWER
# ============================================================

builder.add_conditional_edges(
    "structured",
    route_after_structured,
    {
        "rag": "rag",
        "answer": "answer",
    },
)


# ============================================================
# RAG → ANSWER
# ============================================================

builder.add_edge(
    "rag",
    "answer",
)


# ============================================================
# ANSWER → END
# ============================================================

builder.add_edge(
    "answer",
    END,
)


# ============================================================
# GENERAL → END
# ============================================================

builder.add_edge(
    "general",
    END,
)


# ============================================================
# COMPILE
# ============================================================

graph = builder.compile()