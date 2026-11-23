"""Capture validation shared by Selenium collection and offline evaluation."""


def capture_error(response):
    """Reject empty captures and known UI-only text, not answers mentioning login."""
    text = response.strip()
    if not text:
        return "No assistant response captured."
    ui_lines = {
        "sign in to connect to google apps, create images, and more",
        "sign in", "auto", "gemini is ai and can make mistakes.",
        "about gemini", "subscriptions", "for business",
        "show thinking", "thinking", "just a sec...", "loading...",
        "verify you are human", "checking your browser", "sign in to continue",
    }
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if lines and all(line in ui_lines for line in lines):
        return "Only login, verification, loading, or page chrome was captured."
    return None
