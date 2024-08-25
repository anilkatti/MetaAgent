from typing import TypedDict, Literal, Annotated, Sequence
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from planner import planner_node
from coder import coder_node
from evaluator import evaluator_node
import json
from utils import AgentState

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

workflow.add_node("planner_node", planner_node)
workflow.add_node("coder_node", coder_node)
workflow.add_node("evaluator_node", evaluator_node)

workflow.set_entry_point("planner_node")

workflow.add_conditional_edges(
    "planner_node",
    should_continue,
    {
        "continue": "coder_node",
        "stop": END,
    },
)

workflow.add_edge("coder_node", "evaluator_node")
workflow.add_edge("evaluator_node", "planner_node")

graph = workflow.compile()

messages = [HumanMessage(content="Build a weather agent!")]

graph.invoke({"messages": messages})