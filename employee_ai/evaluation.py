import json
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import graph
from app.observability import get_langfuse_handler


# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = Path(__file__).resolve().parent / "evaluation_results.json"

# Keep this separate from the application's main model if you
# later want to use a cheaper/faster model for evaluation.
JUDGE_MODEL = settings.gemini_model

# If True, a 429 is recorded as QUOTA_EXCEEDED and the test
# continues immediately with the next test.
#
# This is intentionally False for retrying because your current
# free-tier limit is 15 requests/minute.
RETRY_ON_QUOTA = False

# Minimum token length used for simple retrieval-overlap scoring.
MIN_TOKEN_LENGTH = 3


# ============================================================
# Evaluation test cases
# ============================================================

EVALUATION_CASES = [
    # --------------------------------------------------------
    # UNSTRUCTURED
    # --------------------------------------------------------
    {
        "id": "U01",
        "question": "How many annual leave days are employees entitled to per year?",
        "expected_route": "unstructured",
        "expected_answer": "25 annual leave days",
    },
    {
        "id": "U02",
        "question": "How many days of annual leave can be carried forward?",
        "expected_route": "unstructured",
        "expected_answer": "50 days",
    },
    {
        "id": "U03",
        "question": "What are the rules for taking sabbatical leave?",
        "expected_route": "unstructured",
        "expected_answer": (
            "Sabbatical requires at least six months notice, "
            "eligibility after five years, can be taken once in "
            "a continuous five-year period, and has specific pay rules."
        ),
    },
    {
        "id": "U04",
        "question": "What is the maximum monthly mobile reimbursement?",
        "expected_route": "unstructured",
        "expected_answer": "INR 1,500 per month for business use",
    },

    # --------------------------------------------------------
    # STRUCTURED
    #
    # IMPORTANT:
    # Replace these employee-specific questions with names
    # that actually exist in your employee_leave table.
    # --------------------------------------------------------
    {
        "id": "S01",
        "question": "How many days of leave did Lisa Smith take?",
        "expected_route": "structured",
        "expected_answer": "Employee leave record",
    },
    {
        "id": "S02",
        "question": "Which employees are in the Engineering department?",
        "expected_route": "structured",
        "expected_answer": "Engineering employees from the employee_leave table",
    },
    {
        "id": "S03",
        "question": "Who has the most leave remaining?",
        "expected_route": "structured",
        "expected_answer": "The employee with the highest remaining_leaves value",
    },

    # --------------------------------------------------------
    # HYBRID
    #
    # These should require BOTH policy/document information
    # and employee/database information.
    # --------------------------------------------------------
    {
        "id": "H01",
        "question": (
            "What is the annual leave entitlement according to "
            "company policy, and how many leaves does Lisa Smith have remaining?"
        ),
        "expected_route": "hybrid",
        "expected_answer": (
            "25 annual leave days according to company policy, "
            "plus Lisa Smith's remaining leave from the employee database."
        ),
    },
    {
        "id": "H02",
        "question": (
            "What is the company's mobile reimbursement limit, "
            "and how much leave does Lisa Smith have remaining?"
        ),
        "expected_route": "hybrid",
        "expected_answer": (
            "INR 1,500 monthly mobile reimbursement limit, "
            "plus Lisa Smith's remaining leave from the employee database."
        ),
    },
    {
        "id": "H03",
        "question": (
            "What is the annual leave entitlement according to company "
            "policy, and how many leaves does Lisa Smith have remaining?"
        ),
        "expected_route": "hybrid",
        "expected_answer": (
            "25 annual leave days and Lisa Smith's remaining leave."
        ),
    },

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------
    {
        "id": "G01",
        "question": "What is an API?",
        "expected_route": "general",
        "expected_answer": "A general explanation of an API.",
    },
    {
        "id": "G02",
        "question": "What is the capital of France?",
        "expected_route": "general",
        "expected_answer": "Paris",
    },
]


# ============================================================
# Gemini evaluation schema
# ============================================================

class JudgeResult(BaseModel):
    correctness: float = Field(
        ge=0,
        le=1,
        description=(
            "How correct the answer is compared with the expected answer. "
            "1 means fully correct."
        ),
    )

    relevance: float = Field(
        ge=0,
        le=1,
        description=(
            "How directly the answer addresses the user's question. "
            "1 means fully relevant."
        ),
    )

    faithfulness: float = Field(
        ge=0,
        le=1,
        description=(
            "How well the answer is supported by the supplied context. "
            "1 means fully supported."
        ),
    )

    reasoning: str = Field(
        default="",
        description="Short explanation for the three scores.",
    )


# ============================================================
# Gemini client
# ============================================================

client = genai.Client(
    api_key=settings.gemini_api_key,
)


# ============================================================
# Utility functions
# ============================================================

def is_quota_error(exc: Exception) -> bool:
    """
    Detect Gemini quota/rate-limit errors.

    We deliberately check the exception text because the exact
    exception class can vary between Gemini SDK versions.
    """
    message = str(exc).upper()

    return (
        "RESOURCE_EXHAUSTED" in message
        or "429" in message
        or "QUOTA_EXCEEDED" in message
    )


def normalize_text(text: str) -> str:
    """Normalize text for simple lexical comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> set[str]:
    """
    Convert text into normalized tokens.

    Very short words are ignored because they usually don't
    provide useful retrieval evidence.
    """
    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= MIN_TOKEN_LENGTH
    }


def calculate_retrieval_accuracy(
    case: dict[str, Any],
    result: dict[str, Any],
) -> float | None:
    """
    Calculate a lightweight retrieval accuracy score.

    This metric is applicable only when the system was expected
    to retrieve documents.

    If the expected answer contains useful terms and those terms
    occur in the retrieved documents, the score increases.

    This is intentionally a baseline metric. A production-grade
    retrieval evaluation would use labelled relevant documents
    or expected page IDs.
    """
    expected_route = case["expected_route"]

    if expected_route not in {"unstructured", "hybrid"}:
        return None

    documents = result.get("documents", [])

    if not documents:
        return 0.0

    retrieved_text = " ".join(
        document.get("text", "")
        for document in documents
    )

    expected_terms = tokenize(case.get("expected_answer", ""))

    if not expected_terms:
        return 1.0 if retrieved_text.strip() else 0.0

    retrieved_terms = tokenize(retrieved_text)

    matched_terms = expected_terms.intersection(retrieved_terms)

    return len(matched_terms) / len(expected_terms)


def build_context_for_judge(result: dict[str, Any]) -> str:
    """
    Build the evidence supplied to the Gemini evaluator.

    Both RAG documents and SQL-agent output are included.
    """
    sections = []

    sql_result = result.get("sql_agent_result")

    if sql_result:
        sections.append(
            "STRUCTURED DATABASE RESULT:\n"
            f"{sql_result}"
        )

    documents = result.get("documents", [])

    if documents:
        document_sections = []

        for index, document in enumerate(documents, start=1):
            source = document.get("source", "unknown")
            page = document.get("page", "unknown")
            content_type = document.get("content_type", "unknown")
            text = document.get("text", "")

            document_sections.append(
                f"DOCUMENT {index}\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content type: {content_type}\n"
                f"Content:\n{text}"
            )

        sections.append(
            "RETRIEVED DOCUMENT CONTEXT:\n"
            + "\n\n".join(document_sections)
        )

    if not sections:
        return "No external context was retrieved."

    return "\n\n====================\n\n".join(sections)


# ============================================================
# Gemini judge
# ============================================================

def judge_answer(
    case: dict[str, Any],
    result: dict[str, Any],
) -> JudgeResult:
    """
    Ask Gemini to evaluate correctness, relevance and faithfulness
    in ONE API call.

    This is intentionally one call rather than three separate
    calls to reduce Gemini quota consumption.
    """
    question = case["question"]
    expected_answer = case["expected_answer"]
    actual_answer = result.get(
        "answer",
        "No answer was generated.",
    )

    context = build_context_for_judge(result)

    prompt = f"""
You are an evaluation judge for an enterprise QA system.

Evaluate the system's answer using ONLY the information supplied
below.

USER QUESTION:
{question}

EXPECTED ANSWER / GROUND TRUTH:
{expected_answer}

SYSTEM ANSWER:
{actual_answer}

EVIDENCE AVAILABLE TO THE SYSTEM:
{context}

Evaluate three independent dimensions:

1. CORRECTNESS
Does the system answer match the expected answer?
Give:
1.0 = fully correct
0.5 = partially correct
0.0 = incorrect

2. RELEVANCE
Does the answer directly address the user's question?
Give:
1.0 = directly answers the question
0.5 = partially addresses it
0.0 = does not answer it

3. FAITHFULNESS
Is the answer supported by the supplied database result
and/or retrieved document context?
Give:
1.0 = fully supported
0.5 = partially supported
0.0 = unsupported or contradicted

Important:
- Do not use outside knowledge.
- Do not reward an answer merely because it sounds plausible.
- If the evidence does not contain enough information, do not
  assume the missing information.
- Evaluate the answer that was actually generated.
- Return numeric scores between 0 and 1.
"""

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": JudgeResult,
        },
    )

    if not response.parsed:
        raise RuntimeError(
            "Gemini evaluator returned no structured result."
        )

    return response.parsed


# ============================================================
# Run one evaluation case
# ============================================================

def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the QA system and evaluate the resulting answer.
    """
    test_id = case["id"]
    question = case["question"]
    expected_route = case["expected_route"]

    print("=" * 80)
    print(f"Test ID        : {test_id}")
    print(f"Question       : {question}")
    print(f"Expected route : {expected_route}")

    start_time = time.perf_counter()

    try:
        langfuse_handler = get_langfuse_handler()

        result = graph.invoke(
            {"query": question},
            config={
                "callbacks": [langfuse_handler],
                "run_name": f"evaluation-{test_id}",
                "metadata": {
                    "application": "employee-ai",
                    "environment": "evaluation",
                    "test_id": test_id,
                },
            },
        )

        latency = time.perf_counter() - start_time

    except Exception as exc:
        latency = time.perf_counter() - start_time

        if is_quota_error(exc):
            print("STATUS         : QUOTA_EXCEEDED")
            print(f"Latency        : {latency:.2f}s")
            print()

            return {
                "id": test_id,
                "question": question,
                "expected_route": expected_route,
                "status": "QUOTA_EXCEEDED",
                "error": str(exc),
                "latency_seconds": round(latency, 2),
            }

        print("STATUS         : ERROR")
        print(f"Error          : {exc}")
        print(f"Latency        : {latency:.2f}s")
        print()

        return {
            "id": test_id,
            "question": question,
            "expected_route": expected_route,
            "status": "ERROR",
            "error": str(exc),
            "latency_seconds": round(latency, 2),
        }

    actual_route = result.get("route", "unknown")
    answer = result.get(
        "answer",
        "No answer generated.",
    )

    route_correct = (
        actual_route == expected_route
    )

    print(f"Actual route   : {actual_route}")
    print(
        f"Route accuracy : "
        f"{100 if route_correct else 0:.2f}%"
    )

    # --------------------------------------------------------
    # Retrieval accuracy
    # --------------------------------------------------------

    retrieval_accuracy = calculate_retrieval_accuracy(
        case,
        result,
    )

    # --------------------------------------------------------
    # Gemini judge
    # --------------------------------------------------------

    try:
        judge = judge_answer(
            case,
            result,
        )

        correctness = judge.correctness
        relevance = judge.relevance
        faithfulness = judge.faithfulness
        judge_reasoning = judge.reasoning

    except Exception as exc:
        if is_quota_error(exc):
            print(
                "Gemini judge    : QUOTA_EXCEEDED"
            )

            return {
                "id": test_id,
                "question": question,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "route_correct": route_correct,
                "answer": answer,
                "status": "JUDGE_QUOTA_EXCEEDED",
                "latency_seconds": round(latency, 2),
                "retrieval_accuracy": retrieval_accuracy,
                "error": str(exc),
            }

        print(
            f"Gemini judge    : ERROR - {exc}"
        )

        return {
            "id": test_id,
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "answer": answer,
            "status": "JUDGE_ERROR",
            "latency_seconds": round(latency, 2),
            "retrieval_accuracy": retrieval_accuracy,
            "error": str(exc),
        }

    print(
        f"Answer correctness : "
        f"{correctness * 100:.2f}%"
    )
    print(
        f"Answer relevance   : "
        f"{relevance * 100:.2f}%"
    )
    print(
        f"Faithfulness       : "
        f"{faithfulness * 100:.2f}%"
    )

    if retrieval_accuracy is None:
        print(
            "Retrieval accuracy : N/A"
        )
    else:
        print(
            f"Retrieval accuracy : "
            f"{retrieval_accuracy * 100:.2f}%"
        )

    print(
        f"Latency            : "
        f"{latency:.2f}s"
    )

    print(
        f"Answer             : "
        f"{answer}"
    )

    if judge_reasoning:
        print(
            f"Judge reasoning    : "
            f"{judge_reasoning}"
        )

    print()

    return {
        "id": test_id,
        "question": question,
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_correct": route_correct,
        "answer": answer,
        "status": "COMPLETED",
        "answer_correctness": correctness,
        "answer_relevance": relevance,
        "faithfulness": faithfulness,
        "retrieval_accuracy": retrieval_accuracy,
        "latency_seconds": round(latency, 2),
        "judge_reasoning": judge_reasoning,
        "documents_retrieved": len(
            result.get("documents", [])
        ),
    }


# ============================================================
# Metric calculation
# ============================================================

def average(values: list[float]) -> float | None:
    """
    Calculate an average while ignoring unavailable metrics.
    """
    if not values:
        return None

    return sum(values) / len(values)


def calculate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate the five requested evaluation metrics.

    Important:
    - Quota failures are excluded.
    - Retrieval and faithfulness are evaluated only where
      retrieved documents are relevant to the test.
    """

    completed_results = [
        result
        for result in results
        if result.get("status") == "COMPLETED"
    ]

    # --------------------------------------------------------
    # Route Accuracy
    # --------------------------------------------------------

    route_values = [
        1.0 if result["route_correct"] else 0.0
        for result in completed_results
        if "route_correct" in result
    ]

    route_accuracy = average(route_values)

    # --------------------------------------------------------
    # Answer Correctness
    # --------------------------------------------------------

    correctness_values = [
        result["answer_correctness"]
        for result in completed_results
        if "answer_correctness" in result
    ]

    answer_correctness = average(
        correctness_values
    )

    # --------------------------------------------------------
    # Answer Relevance
    # --------------------------------------------------------

    relevance_values = [
        result["answer_relevance"]
        for result in completed_results
        if "answer_relevance" in result
    ]

    answer_relevance = average(
        relevance_values
    )

    # --------------------------------------------------------
    # Faithfulness
    #
    # Only meaningful for tests where the system is expected
    # to retrieve document context.
    # --------------------------------------------------------

    faithfulness_values = [
        result["faithfulness"]
        for result in completed_results
        if (
            result.get("expected_route")
            in {"unstructured", "hybrid"}
            and "faithfulness" in result
        )
    ]

    faithfulness = average(
        faithfulness_values
    )

    # --------------------------------------------------------
    # Retrieval Accuracy
    # --------------------------------------------------------

    retrieval_values = [
        result["retrieval_accuracy"]
        for result in completed_results
        if result.get("retrieval_accuracy") is not None
    ]

    retrieval_accuracy = average(
        retrieval_values
    )

    # --------------------------------------------------------
    # Overall
    #
    # Average the five metric-level scores.
    # Metrics unavailable because of zero applicable cases
    # are excluded.
    # --------------------------------------------------------

    metric_values = [
        value
        for value in [
            route_accuracy,
            answer_correctness,
            answer_relevance,
            faithfulness,
            retrieval_accuracy,
        ]
        if value is not None
    ]

    overall = average(metric_values)

    return {
        "route_accuracy": route_accuracy,
        "answer_correctness": answer_correctness,
        "answer_relevance": answer_relevance,
        "faithfulness": faithfulness,
        "retrieval_accuracy": retrieval_accuracy,
        "overall": overall,
        "completed_tests": len(completed_results),
        "total_tests": len(results),
        "failed_tests": len(results) - len(completed_results),
    }


# ============================================================
# Print final evaluation report
# ============================================================

def print_final_report(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:

    def percentage(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.0f}%"

    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(
        f"Route Accuracy       "
        f"{percentage(metrics['route_accuracy'])}"
    )

    print(
        f"Answer Correctness   "
        f"{percentage(metrics['answer_correctness'])}"
    )

    print(
        f"Answer Relevance     "
        f"{percentage(metrics['answer_relevance'])}"
    )

    print(
        f"Faithfulness         "
        f"{percentage(metrics['faithfulness'])}"
    )

    print(
        f"Retrieval Accuracy   "
        f"{percentage(metrics['retrieval_accuracy'])}"
    )

    print("-" * 60)

    print(
        f"Overall              "
        f"{percentage(metrics['overall'])}"
    )

    print("=" * 60)

    print()
    print(
        f"Completed tests: "
        f"{metrics['completed_tests']}/"
        f"{metrics['total_tests']}"
    )

    print(
        f"Excluded tests: "
        f"{metrics['failed_tests']}"
    )

    print()
    print("Detailed results:")
    print()

    for result in results:

        test_id = result["id"]
        status = result.get("status")

        if status != "COMPLETED":
            print(
                f"{test_id} | STATUS={status}"
            )
            continue

        route_score = (
            100
            if result.get("route_correct")
            else 0
        )

        correctness = (
            result["answer_correctness"] * 100
        )

        relevance = (
            result["answer_relevance"] * 100
        )

        faithfulness = (
            result["faithfulness"] * 100
        )

        retrieval = result.get(
            "retrieval_accuracy"
        )

        retrieval_text = (
            "N/A"
            if retrieval is None
            else f"{retrieval * 100:.0f}%"
        )

        print(
            f"{test_id} | "
            f"Route={route_score:.0f}% | "
            f"Correct={correctness:.0f}% | "
            f"Relevant={relevance:.0f}% | "
            f"Faithful={faithfulness:.0f}% | "
            f"Retrieval={retrieval_text}"
        )


# ============================================================
# Save results
# ============================================================

def save_results(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:

    output = {
        "judge_model": JUDGE_MODEL,
        "metrics": metrics,
        "results": results,
    }

    RESULTS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Evaluation results saved to: "
        f"{RESULTS_FILE}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("Employee AI Evaluation")
    print("=" * 80)
    print()
    print(f"Judge model: {JUDGE_MODEL}")
    print(
        "Gemini judge calls per completed test: 1"
    )
    print(
        "Quota errors will NOT be counted as 0%."
    )
    print()

    results = []

    for case in EVALUATION_CASES:

        result = evaluate_case(case)

        results.append(result)

        # Save after every test so that if the quota is exhausted
        # halfway through the evaluation, previous results are
        # still preserved.
        metrics = calculate_metrics(results)

        save_results(
            results,
            metrics,
        )

    final_metrics = calculate_metrics(
        results
    )

    print_final_report(
        results,
        final_metrics,
    )

    save_results(
        results,
        final_metrics,
    )


if __name__ == "__main__":
    main()