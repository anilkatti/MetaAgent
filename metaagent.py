from typing import TypedDict, Literal, Annotated, Sequence
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from planner import planner_node
from coder import coder_node
from evaluator import evaluator_node
import json
from utils import AgentState, outfile
import sys

MAX_ITERATIONS = 3
TASK = sys.argv[1]

class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

def should_continue(state: AgentState, config: GraphConfig):
    eval_results = state["eval_results"]
    if len(eval_results) == 0:
        return "continue"
    
    if len(eval_results) >= MAX_ITERATIONS:
        return "stop"
    
    last_result = state["eval_results"][-1].content
    if "PASS" in last_result:
        return "stop"
    else:
        return "continue"

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

task = [HumanMessage(content=TASK)]

task_desc = task[0].content

print(f"Crafting an agent for task: \"{task_desc}\" ...")
print()
graph.invoke({"task": task})