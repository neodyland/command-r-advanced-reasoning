from asyncio import (
    create_subprocess_exec,
    sleep,
    get_event_loop,
    FIRST_COMPLETED,
    wait,
    CancelledError,
)
from contextlib import suppress
from subprocess import PIPE


async def __python(code: str):
    res = await create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "-m",
        "125M",
        "-i",
        "googlefan25/lm-python",
        "python",
        "-iq",
        stderr=PIPE,
        stdin=PIPE,
        stdout=PIPE,
    )
    out, err = await res.communicate(bytes(code, "utf8"))
    return f"Stdout: \n{str(out, 'utf8')}\nStderr: \n{str(err, 'utf8')}".strip()


async def python(code: str, timeout=3.0) -> str:
    loop = get_event_loop()
    task_set = set()
    task_set.add(loop.create_task(__python(__trim_code(code))))
    task_set.add(loop.create_task(sleep(timeout)))

    done_first, pending = await wait(task_set, return_when=FIRST_COMPLETED)
    res = ""
    for coro in done_first:
        try:
            res = coro.result()
        except TimeoutError:
            pass

    for p in pending:
        p.cancel()
        with suppress(CancelledError):
            await p
    if len(res) == 0:
        res = None
    return res or "Python process timeout"


def __trim_code(code: str):
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python") :]
    if code.startswith("```py"):
        code = code[len("```py") :]
    if code.startswith("```"):
        code = code[len("```") :]
    if code.endswith("```"):
        code = code[: -len("```")]
    if code.startswith("`"):
        code = code[len("`") :]
    if code.endswith("`"):
        code = code[: -len("`")]
    return code
