from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = get_client()


def get_langfuse_handler() -> CallbackHandler:
    """Create a Langfuse callback handler for LangChain/LangGraph."""
    return CallbackHandler()