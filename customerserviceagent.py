
from typing import TypedDict, Literal

from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model

workflow = StateGraph(AgentState, config_schema=GraphConfig)
    
system_prompt = "classify the user query into predefined categories"

def classify_query(state, config):
    messages = state["query"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"category": [response.content]}

workflow.add_node("classify_query", classify_query)

system_prompt = "retrieve relevant information based on the query category"

def retrieve_information(state, config):
    category = state["category"]
    query = state["query"]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Category: {category}, Query: {query}"}]
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"information": [response.content]}

workflow.add_node("retrieve_information", retrieve_information)

system_prompt = "Generate a customer service response based on the provided information and query"

def generate_response(state, config):
    information = state["information"]
    query = state["query"]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": information}, {"role": "user", "content": query}]
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"response": [response.content]}

workflow.add_node("generate_response", generate_response)

followup_prompt = "determine if a follow-up question is needed"

def check_followup(state, config):
    messages = state["response"]
    messages = [{"role": "system", "content": followup_prompt}] + messages
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"needs_followup": [response.content]}

workflow.add_node("check_followup", check_followup)

workflow.add_edge("classify_query", "retrieve_information")

workflow.add_edge("retrieve_information", "generate_response")

workflow.add_edge("generate_response", "check_followup")

graph = workflow.compile()