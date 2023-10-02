import os
import random
import time
import datetime

import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import wandb
from wandb.sdk.data_types.trace_tree import Trace

import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

PROJECT = "dlai_llm"
MODEL_NAME = "gpt-3.5-turbo"

wandb.login(anonymous="allow")
run = wandb.init(project=PROJECT, job_type="generation")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwards):
    return openai.ChatCompletion.create(**kwards)

def generate_and_print(system_prompt, user_prompt, table, n=5):
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]
    start_time = time.time()
    responses = completion_with_backoff(
        model=MODEL_NAME,
        messages=messages,
        n = n,
        )
    elapsed_time = time.time() - start_time
    for response in responses.choices:
        generation = response.message.content
        print(generation)
    table.add_data(system_prompt,
                user_prompt,
                [response.message.content for response in responses.choices],
                elapsed_time,
                datetime.datetime.fromtimestamp(responses.created),
                responses.model,
                responses.usage.prompt_tokens,
                responses.usage.completion_tokens,
                responses.usage.total_tokens
                )

    # table.add_data(
    #     system_prompt,
    #     user_prompt,
    #     [response.message.content for response in responses.choices],
    #     elapsed_time,
    #     datetime.datetime.fromtimestamp(responses.created),
    #     responses.usage.prompt_tokens,
    #     responses.usage.completion_tokens,
    #     responses.usage.total_tokens
    # )

system_prompt = """You are a creative copywriter.You're given a category of game asset, \
and your goal is to design a name of that asset.The game is set in a fantasy world \
where everyone laughs and respects each other, while celebrating diversity."""

columns = ["system_prompt", "user_prompt", "generations", "timestamp", "elapsed_time", "model", "prompt_tokens", "completion_tokens", "total_tokens"]
table = wandb.Table(columns=columns)

user_prompt = "hero"
generate_and_print(system_prompt, user_prompt, table)

user_prompt = "jewel"
generate_and_print(system_prompt, user_prompt, table)

wandb.log({"simple_generations": table})
run.finish()

worlds = [
    "a mystic medieval island inhabited by intelligent and funny frogs",
    "a modern castle sitting on top of a volcano in a faraway galaxy",
    "a digital world inhabited by friendly machine learning engineers"
]

model_name = "gpt-3.5-turbo"
temperature = 0.7
system_message = """You are a creative copywriter. 
You're given a category of game asset and a fantasy world.
Your goal is to design a name of that asset.
Provide the resulting name only, no additional description.
Single name, max 3 words output, remember!"""

def run_creative_chain(query):
    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    
    root_span = Trace(
        name = "MyCreativeChain",
        kind = "chain",
        start_time_ms = start_time_ms,
        metadata = {"user":"student_1"},
        model_dict= {"_kind":"CreativeChain"}
    )
    
    time.sleep(3)
    world = random.choice(worlds)
    expanded_prompt = f"Game asset category: {query}; fantasy world description: {world}"
    tool_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    
    tool_span = Trace(
        name = "WorldPicker",
        kind = "tool",
        status_code = "success", 
        start_time_ms = start_time_ms,
        end_time_ms = tool_end_time_ms,
        inputs={"input": query},
        outputs={"result": expanded_prompt},
        model_dict={"_kind":"tool", "num_worlds": len(worlds)}
        )
    
    root_span.add_child(tool_span)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": expanded_prompt},
    ]
    
    response = completion_with_backoff(
        model=model_name,
        messages=messages,
        max_tokens=12,
        temperature=temperature)
    
    llm_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    response_text = response["choices"][0]["message"]["content"]
    token_usage = response["usage"].to_dict()
    
    llm_span = Trace(
        name="OpenAI", 
        kind="llm",
        status_code="success",
        metadata={"temperature": temperature, "token_usage": token_usage, "model_name": model_name},
        start_time_ms=tool_end_time_ms,
        end_time_ms=llm_end_time_ms,
        inputs={"system_prompt": system_message, "query": expanded_prompt},
        outputs={"response": response_text},
        model_dict={"_kind":"Openai", "engine":response["model"], "model":response["object"]}
        )
    root_span.add_child(llm_span)
    
    root_span.add_inputs_and_outputs(
        inputs={"query": query},
        outputs={"response": response_text}
    )
    root_span.end_time_ms = llm_end_time_ms
    
    root_span.log(name="creative_trace")
    print(f"Result: {response_text}")
    

wandb.init(project=PROJECT, job_type="generation")
run_creative_chain("hero")
run_creative_chain("jewel")
wandb.finish()

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

from typing import Optional

from langchain.callbacks.manager import(AsyncCallbackManagerForToolRun,
                                        CallbackManagerForToolRun,)

wandb.init(project=PROJECT, job_type="generation")
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

class WorldPickerTool(BaseTool):
    name = "pick_world"
    description = "pick a virtual game world for your character or item naming"
    worlds = [
        "a mystic medieval island inhabited by intelligent and funny frogs",
        "a modern castle sitting on top of a volcano in a faraway galaxy",
        "a digital world inhabited by friendly machine learning engineers"
        ]
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Tool."""
        time.sleep(1)
        return random.choice(self.worlds)
    
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("pick_world does not support async")

class NameValidatorTool(BaseTool):
    name = "validate_name"
    description = "validate if the name is proper generated"
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Tool."""
        time.sleep(1)
        if len(query) < 20:
            return f"This is a correct name:{query}"
        else:
            return f"This name is too long. It should be shorter than 20 characters."
    
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("validate_name does not support async")

llm = ChatOpenAI(temperature=0.7)

tools = [WorldPickerTool(), NameValidatorTool()]
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True
    )

agent.run(
    "Find a virtual game world for me and imagine the name of a hero in that world"
)

agent.run(
    "Find a virtual game world for me and imagine the name of a jewel in that world"
)

agent.run(
    "Find a virtual game world for me and imagine the name of food in that world."
)

wandb.finish()