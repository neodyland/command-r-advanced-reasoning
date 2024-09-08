from openai_api import openai_api, tok
import datetime
from search import search
from access import access
from python import python
import copy

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")


def now():
    return datetime.datetime.now(JST).strftime("%Y/%m/%d")


async def __chat(history: list, q: str, plan: str):
    history = copy.deepcopy(history)
    history.append(
        {
            "role": "system",
            "content": f"""To solve user's query, you should choose from one of the following action:
- "%plan: <short planning>" Plan for answering.
- "%python: <math code>" Run python code for math problems.
- "%search: <query>" Search for the content with the query. Returns list of urls related.
- "%access: <single url>" Access to a single url and extract markdown from it. Returns the content.
- "%think: <short thought>" Think for the content for advanced reasoning. This is your internal thinking, not the final output.
- "%wait: <short thought>" Think deeply to wait a moment to verify or correct your answer. Commonly used after %think.
- "%output: <ultra detailed markdown response>" Final ultra detailed, well formatted markdown response to user with evidence/source to every information.
You could only do one action per response.
Always provide evidence when you %think or %output.
Don't output unnecessary words.
If you followed the instruction correctly, you will be tipped $10.

{f'Your current plan is: {plan}' if plan else ''}
Remember, the current quetion is: {q}""",
        }
    )
    prompt = tok.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    async for x in openai_api(
        prompt,
        {
            "stop": "%",
            "grammar": 'root ::= "%" [^%]+',
        },
    ):
        yield x


def create_history():
    return [
        {
            "role": "system",
            "content": f"""## System Preamble
You are a world-class AI system, capable of advanced reasoning and planning.
Plan for the question using %plan, reason through the query using %think, use "wait a moment... maybe I made a mistake" using %wait, and then provide your final response using %output.

Knowledge Cutoff Date: Online(You can access to the internet through %search action.)
You cannot get correct information without using %search and %access action.

Today's Date: {now()}
""",
        }
    ]


async def chat(input_fn, print_fn, max_search=10, max_python=10):
    h = create_history()
    current_question = None
    accessed = []
    searched = 0
    python_count = 0
    plan = None
    while True:
        last = h[-1]["content"]
        if last.startswith("%search: "):
            if searched > max_search - 1:
                await print_fn(f"Search result: You are searching too much.\n")
                h.append(
                    {
                        "role": "user",
                        "content": f"Search result: You are searching too much.",
                    }
                )
            else:
                searched += 1
                res = await search(last[len("%search: ") :])
                await print_fn(f"Search result: {res}\n")
                h.append({"role": "user", "content": f"Search result: {res}"})
        elif last.startswith("%access: "):
            await print_fn("Access result: ")
            async for x in access(last[len("%access: ") :], current_question, accessed):
                if "all" in x:
                    await print_fn("\n")
                    h.append({"role": "user", "content": f"Access result: {x['all']}"})
                elif "part" in x:
                    await print_fn(x["part"])
            accessed.append(last[len("%access: ") :])
        elif last.startswith("%python: "):
            if python_count > max_python - 1:
                await print_fn(f"Python result: You are using python too much.\n")
                h.append(
                    {
                        "role": "user",
                        "content": f"Python result: You are using python too much.",
                    }
                )
            else:
                python_count += 1
                await print_fn("Python result: ")
                res = await python(last[len("%python: ") :])
                await print_fn(f"{res}\n")
                h.append({"role": "user", "content": f"Python result: {res}"})
        elif last.startswith("%output: ") or len(h) == 1:
            accessed = []
            searched = 0
            python_count = 0
            if isinstance(input_fn, str):
                if len(h) == 1:
                    h.append({"role": "user", "content": input_fn})
                else:
                    return h
            else:
                i = await input_fn()
                if i == "clear":
                    h = create_history()
                    continue
                current_question = i
                h.append({"role": "user", "content": i})
        else:
            if last.startswith("%plan"):
                plan = last[len("%plan: ") :]
            h.append(
                {
                    "role": "system",
                    "content": "think or wait or search or access or plan or output.",
                }
            )
        async for x in __chat(h, current_question, plan):
            if "part" in x:
                await print_fn(x["part"])
            if "all" in x:
                h.append({"role": "assistant", "content": x["all"]})
                await print_fn("\n")


if __name__ == "__main__":
    import asyncio

    async def input_fn():
        return input("Input: ").strip()

    async def print_fn(content):
        print(content, end="", flush=True)

    asyncio.run(chat(input_fn, print_fn))
