"""
Poisson load generator for LangGraph applications.

Generates request traffic following a Poisson process distribution
for load testing purposes.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph


@dataclass
class Request:
    """Holds timing information for a single request."""

    result: Any = None
    start_time: float = 0.0
    end_time: float = 0.0


async def run_request(app: StateGraph, input_data: Dict):
    """Execute a single async request and record timing."""
    start_time = time.time()
    result = await app.ainvoke(input_data)
    end_time = time.time()
    return Request(result=result, start_time=start_time, end_time=end_time)


async def poisson_load_generator(app: StateGraph, inputs: List[Dict], rate: float):
    """
    Send requests to the LangGraph app following a Poisson distribution.

    The time between requests follows an exponential distribution
    with mean 1/rate seconds.

    Args:
        app: The compiled LangGraph application.
        inputs: A list of inputs, where each input is a dictionary.
        rate: The average number of requests per second (lambda).
    """
    logging.info(f"Starting Poisson load generator with a rate of {rate} reqs/sec.")
    # Create a list to hold all the asynchronous tasks
    tasks = []
    for single_input in inputs:
        # Schedule the request to run concurrently
        task = asyncio.create_task(run_request(app, single_input))
        tasks.append(task)

        # Calculate the random time to wait before sending the next request
        # The time between events in a Poisson process is exponentially distributed.
        time_to_wait = random.expovariate(rate)
        await asyncio.sleep(time_to_wait)

    # Wait for all the scheduled tasks to complete and gather their results
    logging.info("All requests sent. Waiting for responses...")
    results = await asyncio.gather(*tasks)
    return results
