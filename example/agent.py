from typing import TypedDict, Literal, Annotated, Sequence

from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage
import json

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    place_name: str
    current_weather: str

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model

system_prompt = """ Be a helpful assistant and help answer the weather question. Your output should be in the JSON format with one of the two keys:

                    1. place_name: name of the city for which the weather needs to be looked up
                    2. final_answer: actual answer based on the weather information that was provided
                """

def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"messages": [response.content]}

def get_weather(state, config):
    messages = [HumanMessage(content="Bright & Sunny")]
    return {"messages": messages}

def should_continue(state, config):
    last_message = state["messages"][-1]
    try:
        message_content = json.loads(last_message.content)
        if "place_name" in message_content:
            return "continue"
        elif "final_answer" in message_content:
            print(message_content["final_answer"])
            return "stop"
        else:
            return "stop"
    except json.JSONDecodeError:
        print("Last message is not a valid JSON")
        return "stop"

class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("call_model", call_model)
workflow.add_node("get_weather", get_weather)

workflow.set_entry_point("call_model")

workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "get_weather",
        "stop": END,
    },
)

workflow.add_edge("get_weather", "call_model")

graph = workflow.compile()

messages = [HumanMessage(content="What's the weather in SF?")]

graph.invoke({"messages": messages})