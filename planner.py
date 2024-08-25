from utils import _get_model
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from typing import List, Annotated
from dataclasses import dataclass
import json
import os

# TODO: match with other nodes
@dataclass
class EvalResult:
    passed: bool
    details: str

# TODO: match with other nodes
@dataclass
class HLAState():
    lla_code: str
    lla_graph: Annotated[dict, "json describing the graph nodes and edges"]
    task: str
    eval_results: List[EvalResult]

RESPONSE_FORMAT ="""
{
    "nodes": [
        {
            "name": str,
            "description": str,
            "input_names": [str],
            "output_names": [str]
        }
    ],
    "edges": [
        {
            "source_node": str,
            "destination_node": str
        }
    ],
    "cond_edges": [
        {
            "source_node": str,
            "destinations": [str]
        }
    ]
}
"""
NEW_PLAN_PROMPT_TEMPLATE = PromptTemplate.from_template(
"""In LangGraph, nodes are typically python functions where the first positional argument is the state.
The START Node is a special node that represents the node sends user input to the graph. The main purpose for referencing this node is to determine which nodes should be called first.
The END Node is a special node that represents a terminal node. This node is referenced when you want to denote which edges have no actions after they are done.

Edges define how the logic is routed and how the graph decides to stop. This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:
- Normal Edges: Go directly from one node to the next.
- Conditional Edges: Call a function to determine which node(s) to go to next.
- Entry Point: Which node to call first when user input arrives.

Describe the LangGraph nodes and edges you would use to accomplishes this task:
{task}

Give your response as JSON in the following format, don't include an explanation or anything other than JSON:
{format}
"""
)
REVISION_PROMPT_TEMPLATE = PromptTemplate.from_template(
"""In LangGraph, nodes are typically python functions where the first positional argument is the state.
The START Node is a special node that represents the node sends user input to the graph. The main purpose for referencing this node is to determine which nodes should be called first.
The END Node is a special node that represents a terminal node. This node is referenced when you want to denote which edges have no actions after they are done.

Edges define how the logic is routed and how the graph decides to stop. This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:
- Normal Edges: Go directly from one node to the next.
- Conditional Edges: Call a function to determine which node(s) to go to next.
- Entry Point: Which node to call first when user input arrives.

Describe the LangGraph nodes and edges you would use to accomplishes this task:
{task}

A previous iteration attempted this, and output this graph:
{graph}

But, when we evaluated the result this is what we found:
{eval_result}

Try again, fixing the previous attempt. Respond with JSON in the following format, don't include an explanation or anything other than JSON:
{format}
"""
)

def planner_node(state: HLAState, config) -> HLAState:
    assert state.task

    llm = _get_model("anthropic")

    if len(state.eval_results) == 0:
        chain = NEW_PLAN_PROMPT_TEMPLATE | llm
        response = chain.invoke({"task": state.task, "format": RESPONSE_FORMAT})
        # TODO: handle invalid json
        state.lla_graph = json.loads(response.content)
        return state
    else:
        chain = REVISION_PROMPT_TEMPLATE | llm
        response = chain.invoke({"task": state.task, "format": RESPONSE_FORMAT, "graph": state.lla_graph, "eval_result": state.eval_results[-1].details})
        # TODO: handle invalid json
        state.lla_graph = json.loads(response.content)
        return state
