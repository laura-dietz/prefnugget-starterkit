import asyncio
import dspy
from pydantic import BaseModel
from typing import List, Optional

class JudgeRelevance(dspy.Signature):
    """Is the response relevant to the query?"""
    query: str = dspy.InputField()
    response: str = dspy.InputField()
    relevant: bool = dspy.OutputField()

class Item(BaseModel):
    query: str
    response: str
    relevant: Optional[bool] = None

async def run_batch(items: List[Item], concurrency=8):
    lm = dspy.LM("openai/gpt-4o-mini")
    sem = asyncio.Semaphore(concurrency)

    with dspy.context(lm=lm):
        predictor = dspy.ChainOfThought(JudgeRelevance)

        async def process(item):
            async with sem:
                # Robust async call
                if hasattr(predictor, "acall"):
                    pred = await predictor.acall(query=item.query, response=item.response)
                else:
                    loop = asyncio.get_running_loop()
                    pred = await loop.run_in_executor(
                        None, lambda: predictor(query=item.query, response=item.response)
                    )
                item.relevant = pred.relevant
                return item

        return await asyncio.gather(*[process(i) for i in items])

items = [Item(query="What is Python?", response="A programming language.")]
results = asyncio.run(run_batch(items))
print(results)