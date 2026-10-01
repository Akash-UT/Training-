import json
from typing import Any, Literal

from google import genai
from langchain.agents import create_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import settings
from app.database import get_langchain_database


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=settings.gemini_api_key
)


# ============================================================
# ROUTING
# ============================================================

class RouteDecision(BaseModel):
    """
    Represents Gemini's semantic routing decision.
    """

    route: Literal[
        "structured",
        "unstructured",
        "hybrid",
        "general",
    ]

    employee_name: str | None = Field(
        default=None,
        description=(
            "Employee name explicitly mentioned in the query. "
            "Do not invent or infer a name."
        ),
    )

    reasoning: str = Field(
        default="",
        description=(
            "Brief explanation of why the selected information "
            "source or sources are required."
        ),
    )


ROUTER_PROMPT = """
You are an intelligent decision-making component for a company
question-answering system.

Your responsibility is to understand the user's question and decide which
available information source should be used to answer it.

The system provides two information sources:

1. EMPLOYEE RECORD TOOL

This tool provides access to the company's structured employee records.

Use this source when the answer depends on information about employees
that is stored in the employee records.

The tool can perform lookups, filtering, calculations, comparisons,
aggregations, and other queries over those employee records.

2. DOCUMENT RETRIEVAL TOOL

This tool provides access to the currently indexed company document.

Use this source when the answer depends on information contained in that
document.

The document may contain any type of company information, including
narrative text, tables, reports, financial information, procedures,
summaries, or other business information.

The document's subject, structure, and content are not known to you
in advance.

Do not assume that the document is an HR policy document.

3. BOTH

Use both information sources only when the question genuinely requires
information from the employee records AND information from the indexed
document to produce an accurate answer.

Do not choose BOTH merely because both sources could contain related
information.

4. GENERAL

Use this when neither company information source is required, such as
casual conversation or general questions unrelated to the available
company information.

DECISION PROCESS:

1. Understand the meaning and intent of the user's question.

2. Determine what factual information is required to answer the question.

3. Determine which available information source is capable of providing
   that information.

4. Select the source or sources that are actually required.

5. Base the routing decision on the meaning of the question and the
   information required, not on individual words.

6. Do not use keyword matching.

7. Do not use hardcoded employee names.

8. Do not use predefined question patterns.

9. Do not use fixed routing rules based on particular words, concepts,
   numbers, dates, or phrases.

10. Do not assume that questions involving numbers, amounts, years,
    calculations, financial information, tables, or comparisons belong
    to the employee record tool.

11. A question about numerical or financial information may belong to the
    document retrieval tool if that information is contained in the
    indexed document.

12. A question involving a year, amount, percentage, calculation, or
    comparison does not by itself determine the route.

13. If the question asks about information that is not part of the
    employee records but is expected to be found in the indexed document,
    select UNSTRUCTURED.

14. If the question asks about information stored in the employee
    records, select STRUCTURED.

15. If answering the question requires both employee-record information
    and information from the indexed document, select HYBRID.

16. If neither source is required, select GENERAL.

The same concept may require different sources depending on the user's
intent and the information required to answer the question.

For example, a question containing a year or a numerical value should
not automatically be classified as STRUCTURED. First determine whether
the requested information belongs to the employee records or to the
indexed document.

EMPLOYEE NAME:

If a specific employee is explicitly mentioned in the user's question,
extract the employee's name when possible.

Do not infer an employee name from context.

If no employee name is explicitly mentioned, return null.

OUTPUT:

Return JSON only.

Return exactly:

{
  "route": "structured | unstructured | hybrid | general",
  "employee_name": "employee name or null",
  "reasoning": "brief explanation of what information is required and why the selected source or sources are appropriate"
}
"""

def route_query(query: str) -> RouteDecision:
    """
    Ask Gemini to semantically determine which information source
    should be used for the user's question.
    """

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            ROUTER_PROMPT,
            f"\nUser query:\n{query}",
        ],
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty routing response."
        )

    try:
        data = json.loads(response.text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON for the routing decision."
        ) from exc

    route = data.get("route")

    if isinstance(route, str):
        data["route"] = route.strip().lower()

    return RouteDecision.model_validate(data)


# ============================================================
# LANGCHAIN SQL AGENT
# ============================================================

sql_llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.gemini_api_key,
)


sql_db = get_langchain_database()


sql_toolkit = SQLDatabaseToolkit(
    db=sql_db,
    llm=sql_llm,
)


sql_tools = sql_toolkit.get_tools()


SQL_AGENT_PROMPT = """
You are a PostgreSQL database agent for a company question-answering
system.

Your job is to answer questions using the structured employee information
available through the database tools.

Instructions:

1. Understand the user's question and determine what information is
   required to answer it.

2. Use the available database tools to inspect the database schema when
   necessary.

3. Generate an appropriate SQL query based on the user's question and
   the discovered database schema.

4. Execute the SQL query using the available database tools.

5. Base your answer only on the actual database results.

6. Never invent employee names, values, rows, calculations, or results.

7. Only perform read-only database operations.

8. Never perform operations that modify the database, including:

   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   TRUNCATE
   CREATE
   GRANT
   REVOKE
   MERGE

9. When filtering text values, use appropriate case-insensitive matching
   when necessary.

10. For questions requiring filtering, aggregation, calculations,
    comparisons, counts, averages, totals, minimums, or maximums,
    construct the appropriate SQL query rather than assuming the result.

11. Never assume that information exists in the database simply because
    the user asks about it.

12. Inspect the available schema and query only information that is
    actually available in the database.

13. If the database does not contain enough information to answer the
    question, clearly state that the available database information is
    insufficient.

14. Preserve the values returned by the database accurately.

15. Do not use outside knowledge to fill missing database information.

16. Return a concise natural-language answer based only on the database
    result.
"""


sql_agent = create_agent(
    model=sql_llm,
    tools=sql_tools,
    system_prompt=SQL_AGENT_PROMPT,
)


def run_sql_agent(query: str) -> str:
    """
    Execute a natural-language question against PostgreSQL
    using the LangChain SQL agent.
    """

    result = sql_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    messages = result.get("messages", [])

    if not messages:
        return "No database result was returned."

    content = messages[-1].content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                text = block.get("text")

                if text:
                    text_parts.append(text)

        if text_parts:
            return "\n".join(text_parts).strip()

    return str(content)


# ============================================================
# DOCUMENT-GROUNDED ANSWER
# ============================================================

DOCUMENT_RAG_PROMPT = """
You are a document-grounded QA agent.

Answer the user's question using ONLY the retrieved content from the
currently indexed company document.

Rules:

1. Understand the user's intent before answering.

2. Use all relevant retrieved chunks when information is spread across
   multiple sections, pages, or tables.

3. Retrieved content may contain normal document text or extracted
   table content.

4. Do not assume the document is a policy document or belongs to any
   particular subject.

5. Never use outside knowledge or invent missing information.

6. Preserve names, dates, numbers, units, limits, conditions, and
   terminology exactly as supported by the retrieved document.

7. For tables, correctly relate values to their headers when reliable
   headers are available.

8. If a table has no reliable headers, do not guess or invent the meaning
   of the columns.

9. Preserve important rules, conditions, exceptions, and limitations.

10. Simple calculations are allowed only when all required values come
    from the retrieved document.

11. If the retrieved content is insufficient to answer the question,
    clearly say so.

12. If relevant source metadata is available, mention the page or section.

13. Give a concise and direct answer.

Before answering, verify that every factual statement is supported by
the retrieved document.

Return only the final answer.
"""


# ============================================================
# FINAL ANSWER
# ============================================================

ANSWER_PROMPT = """
You are a company question-answering assistant.

Answer the user's question using ONLY the information provided by the
available company information sources.

The available sources may include:

1. Structured company data from the database.
2. Retrieved content from the currently indexed company document.

Rules:

1. Answer only from the provided information.

2. Do not use outside knowledge to fill missing information.

3. Never invent names, values, dates, calculations, facts, or conclusions.

4. Treat database results as factual structured data.

5. Treat retrieved document content as the source for information
   contained in the indexed document.

6. When both sources are provided, combine them only when both are
   relevant to the user's question.

7. Do not assume that information from one source exists in the other
   source.

8. Preserve names, dates, numbers, units, conditions, limits, exceptions,
   and terminology accurately.

9. If the available information is insufficient, clearly state that the
   available information is insufficient.

10. If the sources contain conflicting information, do not silently
    choose one.

    Clearly identify the conflict and indicate which source contains
    each piece of information.

11. Keep the answer concise and directly address the user's question.

12. Do not mention internal routing, agents, prompts, tools, or system
    implementation details unless explicitly asked by the user.

Before answering, verify that the answer is fully supported by the
provided information.

User question:
{query}

Structured database result:
{sql_agent_result}

Retrieved document content:
{documents}
"""


def generate_answer(
    query: str,
    documents: list[dict[str, Any]] | None = None,
    sql_agent_result: str | None = None,
) -> str:
    """
    Generate the final answer using Gemini based only on the
    information retrieved from the available sources.
    """

    documents = documents or []
    sql_agent_result = sql_agent_result or ""

    prompt = ANSWER_PROMPT.replace(
        "{query}",
        query,
    )

    prompt = prompt.replace(
        "{sql_agent_result}",
        sql_agent_result,
    )

    prompt = prompt.replace(
        "{documents}",
        json.dumps(
            documents,
            default=str,
            indent=2,
        ),
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config={
            "temperature": 0.2,
        },
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty answer."
        )

    return response.text.strip()