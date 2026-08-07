"""Command-line interface.

    name-screening build     [--source data/watchlist_seed.csv]
    name-screening screen    --name "طارق الهاشمي" [--top-k N] [--threshold F]
    name-screening evaluate  [--sweep]
    name-screening serve     [--host H] [--port P]

Also runnable as ``python -m name_screening.cli`` per the brief's acceptance
criteria.

Windows note: the console has to be on a UTF-8 code page or Arabic output comes
back as question marks. Worth handling here rather than leaving a reviewer to
conclude the matching is broken when only the printing is.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="Cross-script Arabic/English name screening for sanctions and PEP lists.",
)


@app.command()
def build(
    source: str = typer.Option(None, help="Watchlist file. Defaults to the committed seed CSV."),
    force: bool = typer.Option(False, help="Rebuild even if an index already exists."),
) -> None:
    """Embed the watchlist and write the FAISS index and metadata sidecar."""
    raise NotImplementedError


@app.command()
def screen(
    name: str = typer.Option(..., help="Name to screen, in either script."),
    top_k: int = typer.Option(None, help="Candidates to retrieve."),
    threshold: float = typer.Option(None, help="Override the configured match threshold."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Screen one name and print ranked candidates."""
    raise NotImplementedError


@app.command()
def evaluate(
    sweep: bool = typer.Option(False, help="Also emit the precision/recall curve."),
) -> None:
    """Run the labelled evaluation set and write the report."""
    raise NotImplementedError


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Run the FastAPI service."""
    raise NotImplementedError


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
