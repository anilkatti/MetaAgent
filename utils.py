from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal, Annotated, Sequence, List
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    task: str
    lla_code: str
    lla_nodes: List[str]
    lla_edges: List[str]
    lla_cond_edges: List[str]
    eval_results: Annotated[Sequence[BaseMessage], add_messages]

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model