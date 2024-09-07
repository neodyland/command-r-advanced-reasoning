from duckduckgo_search import AsyncDDGS
import random


async def search(text: str) -> str:
    res = await AsyncDDGS().atext(
        keywords=text, safesearch="off", max_results=5, backend="html"
    )
    random.shuffle(res)
    return "\n".join([x["href"] for x in res])
