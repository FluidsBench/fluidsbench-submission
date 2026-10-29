"""Current hosting for immutable scientific assets; payload identities are unchanged."""

OLD_RELEASE_BASE = "https://github.com/neilashton/fluidsbench-submission/releases/download/"
RELEASE_BASE = "https://github.com/FluidsBench/fluidsbench-submission/releases/download/"
RELEASE_TAGS = {
    "ahmedml-derived-cache-v1-dev": "ahmedml-derived-cache-v1",
    "drivaernetplusplus-support-v1-dev": "drivaernetplusplus-support-v1",
}


def support_url(url: str) -> str:
    """Resolve only the exact retired release host; never change digest bindings."""
    if not url.startswith(OLD_RELEASE_BASE):
        return url
    tag, separator, asset = url[len(OLD_RELEASE_BASE):].partition("/")
    if not separator or not asset or "?" in url or "#" in url:
        raise ValueError("support URL must name an immutable release asset")
    return RELEASE_BASE + RELEASE_TAGS.get(tag, tag) + "/" + asset
