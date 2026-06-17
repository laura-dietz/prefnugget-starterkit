import asyncio
import dspy
from pydantic import BaseModel
from typing import List, Optional

from minima_llm import OpenAIMinimaLlm
from minima_llm.dspy_adapter import run_dspy_batch

class JudgeRelevance(dspy.Signature):
    """Is the response relevant to the query?"""
    query: str = dspy.InputField()
    response: str = dspy.InputField()
    relevant: bool = dspy.OutputField()

class Item(BaseModel):
    query: str
    response: str
    relevant: Optional[bool] = None

async def run_batch(items: List[Item]):
    backend = OpenAIMinimaLlm.from_env()

    results = await run_dspy_batch(
        signature_class=JudgeRelevance,
        annotation_objs=items,
        output_converter=lambda pred, item: setattr(item, 'relevant', pred.relevant),
        backend=backend,
    )

    await backend.aclose()
    return results

items = [Item(query="What is Python?", response="A programming language.")]
results = asyncio.run(run_batch(items))
print(results)