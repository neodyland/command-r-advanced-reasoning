from openai_api import openai_api, tok
from playwright.async_api import async_playwright
from trafilatura import extract


async def access(url: str, q: str, previous_accessed: list):
    if "\n" in url:
        url = url.split("\n")[0]
    if url in previous_accessed:
        yield {
            "part": "You have accessed this url previously. Try another url. You got no information from this url this time."
        }
        yield {
            "all": "You have accessed this url previously. Try another url. You got no information from this url this time."
        }
        return
    try:
        async with async_playwright() as playwright:
            chromium = playwright.chromium
            browser = await chromium.launch(channel="chrome")
            page = await browser.new_page(
                extra_http_headers={"User-Agent": "NeodyBot/1 https://neody.land/"}
            )
            await page.goto(url, timeout=10.0 * 1000.0)
            await page.wait_for_load_state("load", timeout=10.0 * 1000.0)
            try:
                await page.wait_for_load_state("networkidle", timeout=5.0 * 1000.0)
            except:
                pass
            text = await page.inner_html("html")
            await browser.close()
        text = extract(text, output_format="markdown")
        if text is None or text.strip() == "":
            yield {
                "part": "No information found on this document. This document is empty. You got no information from this url."
            }
            yield {
                "all": "No information found on this document. This document is empty. You got no information from this url."
            }
            return
        async for x in __extract_ai(
            text,
            f"""Extract the information necessary(detailed) from the previous document to answer the following question.
If the document isn't related to the question, start with "No related information found on this document. This document is about"
If the document is realted to the question with confidence, start with "Related information found. This document describes"
If you followed the instruction correctly, you will be tipped $10.

Question: {q}""",
        ):
            yield x
    except Exception:
        yield {
            "part": "Access failed. Try another url. Don't try this url again. You got no information from this url."
        }
        yield {
            "all": "Access failed. Try another url. Don't try this url again. You got no information from this url."
        }


async def __extract_ai(content: str, system: str):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}, {"role": "system", "content": system}],
        tokenize=False,
        add_generation_prompt=False,
    )
    async for x in openai_api(prompt):
        yield x
