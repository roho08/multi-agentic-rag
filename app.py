from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState
from utils.utils import config, new_uuid
from subgraph.graph_builder import researcher_graph
from main_graph.graph_builder import InputState, graph
from langgraph.types import Command
import asyncio
import uuid
#!/usr/bin/env python3

import asyncio
import time
import builtins

thread = {"configurable": {"thread_id": new_uuid()}}
#This is a question related to environmental context. tell me the data center PUE efficiency value in Dublin in 2021

async def process_query(query):
    inputState = InputState(messages=query)

    async for c, metadata in graph.astream(input=inputState, stream_mode="messages", config=thread):
        if c.additional_kwargs.get("tool_calls"):
            print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="", flush=True)
        if c.content:
            time.sleep(0.05)
            print(c.content, end="", flush=True)

    if len(graph.get_state(thread)[-1]) > 0:
        if len(graph.get_state(thread)[-1][0].interrupts) > 0:
            response = input("\nThe response may contain uncertain information. Retry the generation? If yes, press 'y': ")
            if response.lower() == 'y':
                async for c, metadata in graph.astream(Command(resume=response), stream_mode="messages", config=thread):
                    if c.additional_kwargs.get("tool_calls"):
                        print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="")
                    if c.content:
                        time.sleep(0.05)
                        print(c.content, end="", flush=True)


async def main():
    input = builtins.input
    print("Enter your query (type '-q' to quit):")
    while True:
        query = input("> ")
        if query.strip().lower() == "-q":
            print("Exiting...")
            break
        await process_query(query)


if __name__ == "__main__":
    asyncio.run(main())
