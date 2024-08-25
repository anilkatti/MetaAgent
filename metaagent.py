from typing import TypedDict, Literal, Annotated, Sequence
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from planner import planner_node
from coder import coder_node
from evaluator import evaluator_node
import json
from utils import AgentState

def should_continue(state, config):
    eval_results = state["eval_results"]
    if len(eval_results) == 0:
        return "continue"
    
    last_result = state["eval_results"][-1].content
    if "PASS" in last_result:
        return "stop"
    else:
        return "continue"
    
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

task = [HumanMessage(content="Build a weather agent!")]
graph.invoke({"task": task})