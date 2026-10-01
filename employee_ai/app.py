import streamlit as st

from app.graph import graph
from ingest_policy import ingest_document
from sync_documents import (
    sync_document,
    save_hash,
    get_current_hash,
)


st.set_page_config(
    page_title="Employee AI Assistant",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# Document synchronization
# =========================================================

document_changed = sync_document()

if document_changed:

    st.warning(
        "📄 Document changes detected. "
        "Updating Qdrant..."
    )

    with st.spinner(
        "Updating document embeddings and Qdrant..."
    ):
        try:
            # Re-ingest the changed document
            ingest_document()

            # Save the new hash only after
            # successful Qdrant ingestion
            save_hash(
                get_current_hash()
            )

            st.success(
                "✅ Document updated successfully. "
                "Qdrant is now synchronized."
            )

        except Exception as exc:
            st.error(
                f"❌ Document update failed: `{exc}`"
            )

else:

    st.markdown(
    """
    <div style="
        text-align: left;
        font-size: 48px;
        font-weight: 700;
        margin-top: 40px;
        margin-bottom: 30px;
    ">
        Hello !!!
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Application UI
# =========================================================

st.title("AI Assistant  🤖 ")

st.caption("Ask questions. Get answers from your data.")


# =========================================================
# Initialize chat history
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# Display previous messages
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("metadata")
        ):

            metadata = message["metadata"]

            with st.expander(
                "🔍 Execution details"
            ):

                st.write(
                    "**Route:**",
                    metadata.get("route"),
                )

                st.write(
                    "**Reasoning:**",
                    metadata.get("reasoning"),
                )


# =========================================================
# Chat input
# =========================================================

question = st.chat_input(
    "Ask an HR question..."
)


if question:

    # -----------------------------------------------------
    # Display user question
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------------------------------
    # Run LangGraph
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = graph.invoke(
                    {
                        "query": question,
                    }
                )

                answer = result.get(
                    "answer",
                    "I couldn't generate an answer.",
                )

                route = result.get(
                    "route",
                    "unknown",
                )

                reasoning = result.get(
                    "reasoning",
                    "",
                )

                # Display answer
                st.markdown(answer)

                # Execution details
                with st.expander(
                    "🔍 Execution details"
                ):

                    st.write(
                        "**Route:**",
                        route,
                    )

                    st.write(
                        "**Reasoning:**",
                        reasoning,
                    )

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "route": route,
                            "reasoning": reasoning,
                        },
                    }
                )

            except Exception as exc:

                error_message = (
                    f"Something went wrong: `{exc}`"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("System")

    st.write(
        "**Structured data:** PostgreSQL"
    )

    st.write(
        "**Document data:** Qdrant + HR PDF"
    )

    st.write(
        "**LLM:** Gemini"
    )

    st.write(
        "**Orchestration:** LangGraph"
    )

    st.write(
        "**SQL:** LangChain SQL Agent"
    )

    st.divider()

    if st.button("Clear conversation"):

        st.session_state.messages = []

        st.rerun()