from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    url: str
    title: str = ""
    first_seen_step: int = 0
    fetch_count: int = 0
    snippet: str = ""


@dataclass
class SourceTracker:
    """Track consulted sources during a deepdive run."""

    sources: dict[str, Source] = field(default_factory=dict)

    def record_search_result(
        self,
        *,
        url: str,
        title: str,
        snippet: str,
        step: int,
    ) -> None:
        existing = self.sources.get(url)
        if existing is None:
            self.sources[url] = Source(
                url=url,
                title=title,
                snippet=snippet,
                first_seen_step=step,
            )
            return

        if not existing.title and title:
            existing.title = title
        if not existing.snippet and snippet:
            existing.snippet = snippet

    def record_fetch(
        self,
        *,
        url: str,
        step: int,
    ) -> None:
        existing = self.sources.get(url)
        if existing is None:
            self.sources[url] = Source(
                url=url,
                first_seen_step=step,
                fetch_count=1,
            )
            return
        existing.fetch_count += 1

    def numbered_sources(self) -> list[tuple[int, Source]]:
        ordered = sorted(
            self.sources.values(),
            key=lambda source: (source.first_seen_step, source.url),
        )
        return list(enumerate(ordered, start=1))
