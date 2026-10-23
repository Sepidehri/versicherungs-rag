"""Command-line interface: ``vrag ingest | ask | models | eval | eval-retrieval | serve``."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Annotated

import typer

from versicherungs_rag.components import Components
from versicherungs_rag.config import RetrievalMode, get_settings

app = typer.Typer(help="RAG-Assistent für Versicherungsbedingungen mit Modellvergleich.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    # Windows consoles default to cp1252, which cannot print "–", "✓" or "€".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _components() -> Components:
    return Components(get_settings())


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@app.command()
def ingest(
    chunking: Annotated[str | None, typer.Option(help="section | fixed")] = None,
) -> None:
    """Load documents, chunk, embed and save the search index."""
    comps = _components()
    typer.echo(f"Embedding model: {comps.settings.embedding_model}")
    index = comps.build_index(chunking)
    docs = sorted({c.doc_id for c in index.chunks})
    typer.echo(f"Indexed {len(index.chunks)} chunks from {len(docs)} documents -> {comps.settings.index_dir}")


@app.command()
def ask(
    question: str,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    mode: Annotated[RetrievalMode | None, typer.Option(help="Retrieval mode")] = None,
    show_context: Annotated[bool, typer.Option("--context", help="Print retrieved excerpts")] = False,
) -> None:
    """Ask a single question."""
    comps = _components()
    llm = comps.models.client(model or comps.settings.default_model)
    available, status = llm.is_available()
    if not available:
        raise typer.BadParameter(f"{llm.name}: {status}")
    result = comps.pipeline(mode).answer(question, llm)
    typer.echo(f"\n{result.answer}\n")
    for c in result.citations:
        typer.echo(f"  [{c.number}] {c.label}")
    typer.echo(
        f"\n({llm.name}, retrieval {result.retrieval_latency_s * 1000:.0f} ms, "
        f"generation {result.generation_latency_s:.1f} s)"
    )
    if show_context:
        for rc in result.contexts:
            typer.echo(f"\n--- [{rc.rank}] {rc.chunk.citation_label} (score {rc.score:.3f})\n{rc.chunk.text}")


@app.command()
def models() -> None:
    """List configured models and whether they are ready to use."""
    registry = _components().models
    for name in registry.names:
        spec = registry.spec(name)
        available, status = registry.client(name).is_available()
        mark = "✓" if available else "✗"
        typer.echo(f"{mark} {name:<22} {spec.provider:<11} {status if not available else spec.description}")


@app.command("eval")
def evaluate(
    models_: Annotated[str | None, typer.Option("--models", help="Comma-separated; default: all available")] = None,
    mode: Annotated[RetrievalMode | None, typer.Option(help="Retrieval mode")] = None,
    judge: Annotated[str | None, typer.Option(help="Model name to use as faithfulness judge")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions (smoke test)")] = None,
) -> None:
    """Compare LLMs on the test set (same retrieved context for every model)."""
    from versicherungs_rag.evaluation import FaithfulnessJudge, evaluate_models, load_testset, summarize
    from versicherungs_rag.evaluation.report import model_table, write_generation_report
    from versicherungs_rag.prompts import PROMPT_VERSION

    comps = _components()
    s = comps.settings
    cases = load_testset(s.testset_path)[:limit]
    registry = comps.models

    names = models_.split(",") if models_ else registry.names
    clients = []
    for name in names:
        client = registry.client(name.strip())
        available, status = client.is_available()
        if available:
            clients.append(client)
        else:
            typer.echo(f"Skipping {name}: {status}", err=True)
    if not clients:
        raise typer.Exit("No available models to evaluate.")

    judge_client = None
    if judge:
        judge_client = FaithfulnessJudge(registry.client(judge))

    mode = mode or s.retrieval_mode
    pipeline = comps.pipeline(mode)
    typer.echo(f"Evaluating {len(clients)} model(s) on {len(cases)} questions, retrieval={mode.value}")
    results = evaluate_models(cases, pipeline, clients, judge_client, progress=lambda m: typer.echo(f"  {m}"))
    summaries = summarize(results)

    out_dir = s.results_dir / f"eval-{_timestamp()}"
    config = {
        "Fragen": len(cases),
        "Retrieval": mode.value,
        "Top-k": s.top_k,
        "Embedding": s.embedding_model,
        "Reranker": s.reranker_model if mode == RetrievalMode.HYBRID_RERANK else "–",
        "Chunking": comps.index.chunker_name,
        "Judge": judge or "–",
        "Prompt": PROMPT_VERSION,
    }
    report = write_generation_report(out_dir, results, summaries, config)
    typer.echo("\n" + model_table(summaries))
    typer.echo(f"\nReport: {report}")


@app.command("eval-retrieval")
def evaluate_retrieval_cmd(
    embedding_models: Annotated[
        str | None, typer.Option(help="Comma-separated extra embedding models to compare")
    ] = None,
) -> None:
    """Compare chunking strategies, retrieval modes and embedding models (no LLM needed)."""
    from versicherungs_rag.evaluation import evaluate_retrieval, load_testset
    from versicherungs_rag.evaluation.report import retrieval_table, write_retrieval_report
    from versicherungs_rag.ingestion import build_chunker, load_directory
    from versicherungs_rag.retrieval import Retriever, SearchIndex, build_embedder

    comps = _components()
    s = comps.settings
    cases = load_testset(s.testset_path)
    documents = load_directory(s.documents_dir)
    summaries = []

    embedders = [comps.embedder] + [build_embedder(m.strip()) for m in (embedding_models or "").split(",") if m.strip()]
    for embedder in embedders:
        for chunking in ("fixed", "section"):
            chunker = build_chunker(chunking, s.chunk_max_chars, s.chunk_overlap_chars)
            index = SearchIndex.build(documents, chunker, embedder)
            for mode in RetrievalMode:
                if mode == RetrievalMode.BM25 and embedder is not embedders[0]:
                    continue  # BM25 does not depend on the embedding model
                retriever = Retriever(
                    index,
                    embedder,
                    mode,
                    reranker=comps.reranker if mode == RetrievalMode.HYBRID_RERANK else None,
                    candidates=s.rerank_candidates,
                )
                embed_label = "" if mode == RetrievalMode.BM25 else f" · {embedder.name.split('/')[-1]}"
                name = f"{chunking} · {mode.value}{embed_label}"
                typer.echo(f"  {name}")
                summaries.append(evaluate_retrieval(name, cases, retriever, s.top_k))

    out_dir = s.results_dir / f"retrieval-{_timestamp()}"
    config = {"Fragen (beantwortbar)": summaries[0].num_cases, "Top-k": s.top_k, "Reranker": s.reranker_model}
    report = write_retrieval_report(out_dir, summaries, s.top_k, config)
    typer.echo("\n" + retrieval_table(summaries, s.top_k))
    typer.echo(f"\nReport: {report}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the REST API and web UI."""
    import uvicorn

    from versicherungs_rag.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
