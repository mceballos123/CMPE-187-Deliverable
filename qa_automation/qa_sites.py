"""Page details for the four chatbot websites.

Chat UIs change often. If a site stops working, run `python qa_runner.py --probe`,
inspect the page in Chrome DevTools and put new CSS selectors in sites_override.json
(same keys as below), for example:

    {"claude": {"response": ["div.font-claude-response"]}}

No code change is needed. If no response selector matches, the runner falls back to
reading the page text after the prompt ("body-diff" extraction) and records that.
"""

import json
from pathlib import Path

SITES = {
    # Selectors merged with Martin's llm_memory_test.py (Intelligent Memory) so both scripts
    # see the same page elements.
    "chatgpt": {
        "name": "ChatGPT",
        # temporary chat: memory off, chat not saved (matches context class QA-CX-01)
        "url": "https://chatgpt.com/?temporary-chat=true",
        "input": ["#prompt-textarea", "div#prompt-textarea[contenteditable='true']",
                  "div.ProseMirror[contenteditable='true']", "textarea"],
        "response": ["li[data-message-role='assistant']", "div[data-message-author-role='assistant']"],
        "busy": ["button[data-testid='stop-button']", "button[aria-label*='Stop']"],
        "setup": "Temporary chat is used, so memory is off automatically.",
    },
    "claude": {
        "name": "Claude",
        "url": "https://claude.ai/new",
        "input": ["div[contenteditable='true'].ProseMirror", "div[contenteditable='true'][role='textbox']",
                  "div[contenteditable='true']"],
        "response": ["div.font-claude-response", "div[data-testid='assistant-message']",
                     "div.font-claude-message"],
        "busy": ["button[aria-label='Stop response']", "[data-is-streaming='true']", "button[aria-label*='Stop']"],
        "setup": "Turn memory off in Settings before the run.",
    },
    "gemini": {
        "name": "Gemini",
        "url": "https://gemini.google.com/app",
        "input": ["div.ql-editor[contenteditable='true']", "rich-textarea div[contenteditable='true']",
                  "div[contenteditable='true'][role='textbox']", "div[contenteditable='true']"],
        "response": ["message-content", "div.model-response-text", "model-response"],
        "busy": ["button[aria-label='Stop response']", "button.stop", "mat-icon[fonticon='stop']",
                 "button[aria-label*='Stop']"],
        "setup": "Turn off Gemini Apps Activity / personal context before the run.",
    },
    "perplexity": {
        "name": "Perplexity",
        "url": "https://www.perplexity.ai/",
        "input": ["div#ask-input[contenteditable='true']", "#ask-input", "textarea[placeholder]",
                  "div[contenteditable='true'][role='textbox']", "div[contenteditable='true']"],
        "response": ["div[id^='markdown-content']", "div.prose"],
        "busy": ["button[aria-label='Stop']", "button[aria-label='Stop generating']",
                 "button[data-testid='stop-generating-response-button']"],
        "setup": "Use the default search mode. Memory/personalization off in settings.",
    },
}

# Text that means the run cannot continue on its own (checked only when no answer came back)
BLOCK_PATTERNS = {
    "challenge": r"verify you are human|just a moment|unusual activity|captcha|are you a robot",
    "login": r"\blog ?in\b|\bsign ?in\b|\bsign up\b|create (a free )?account",
    "limit": r"usage limit|limit reached|reached (your|the) (limit|message limit)|rate limit|too many requests|"
             r"out of free|try again (later|in)|upgrade to (plus|pro)",
}

# A chatbot's own error message instead of an answer: the execution is an error and is retried
SITE_ERROR = r"^(sorry, )?(something went wrong|an error occurred|there was an error|network error|" \
             r"i'?m having trouble|please try (your request )?again)"


def is_site_error(answer):
    import re
    a = (answer or "").strip().lower()
    return len(a.split()) < 30 and re.search(SITE_ERROR, a) is not None


# Lines of page chrome that can end up in a body-diff answer
BOILERPLATE = {"copy", "copy code", "retry", "share", "edit", "good response", "bad response", "regenerate",
               "read aloud", "sources", "related", "ask follow-up", "rewrite", "show drafts", "thumbs up",
               "thumbs down", "more", "bash", "sh", "shell", "plaintext", "text", "code"}


def load_sites(base_dir):
    sites = json.loads(json.dumps(SITES))
    override = Path(base_dir) / "sites_override.json"
    if override.exists():
        for app, fields in json.loads(override.read_text(encoding="utf-8")).items():
            sites.setdefault(app, {}).update(fields)
    return sites
