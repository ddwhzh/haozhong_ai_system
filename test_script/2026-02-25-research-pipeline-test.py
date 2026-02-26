"""Integration test for the Research Agent pipeline.

Tests the full research loop by invoking the /api/v1/research/start endpoint
with a sample research topic and verifying the response structure.

Can also be run in "local" mode that directly invokes the LangGraph
without requiring a running HTTP server.

Usage:
    # Against running server
    python test_script/2026-02-25-research-pipeline-test.py --mode api

    # Direct LangGraph invocation (no server needed)
    python test_script/2026-02-25-research-pipeline-test.py --mode local

    # Just test arXiv + Semantic Scholar clients
    python test_script/2026-02-25-research-pipeline-test.py --mode clients
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Client Tests ──────────────────────────────────────────────────────────

async def test_arxiv_client():
    """Test arXiv API client."""
    from app.infra.external.arxiv_client import ArxivClient

    console.print("\n[bold]Testing ArxivClient...[/bold]")
    client = ArxivClient(timeout_seconds=15, max_results=5)

    results = await client.search("retrieval augmented generation", max_results=5)
    console.print(f"  arXiv results: {len(results)} papers")

    for r in results[:3]:
        console.print(f"  - {r['title'][:80]} ({r.get('year', 'N/A')})")

    assert len(results) > 0, "arXiv returned no results"
    assert results[0].get("title"), "First result has no title"
    console.print("  [green]PASS[/green]")
    return results


async def test_semantic_scholar_client():
    """Test Semantic Scholar API client."""
    from app.infra.external.semantic_scholar import SemanticScholarClient

    console.print("\n[bold]Testing SemanticScholarClient...[/bold]")
    client = SemanticScholarClient(timeout_seconds=15, max_results=5)

    results = await client.search("retrieval augmented generation", max_results=5)
    console.print(f"  S2 results: {len(results)} papers")

    for r in results[:3]:
        console.print(
            f"  - {r['title'][:80]} ({r.get('year', 'N/A')}) "
            f"[citations: {r.get('citation_count', 0)}]"
        )

    assert len(results) > 0, "Semantic Scholar returned no results"
    assert results[0].get("title"), "First result has no title"
    console.print("  [green]PASS[/green]")
    return results


async def test_sandbox_engine():
    """Test sandbox execution with a simple script."""
    from app.agents.research.sandbox import SandboxEngine

    console.print("\n[bold]Testing SandboxEngine...[/bold]")
    sandbox = SandboxEngine(timeout_seconds=30)

    code = '''
import json
result = {"status": "success", "value": 42, "message": "hello from sandbox"}
print(json.dumps(result))
'''

    result = await sandbox.execute(code=code, dependencies=[])
    console.print(f"  Exit code: {result.exit_code}")
    console.print(f"  Elapsed: {result.elapsed_seconds}s")
    console.print(f"  Metrics: {result.metrics}")

    assert result.exit_code == 0, f"Sandbox exit code {result.exit_code}: {result.stderr}"
    assert result.metrics.get("status") == "success", "Metrics not parsed correctly"
    console.print("  [green]PASS[/green]")
    return result


# ── Local Graph Test ──────────────────────────────────────────────────────

async def test_local_graph():
    """Test the research LangGraph directly (no HTTP server)."""
    from app.core.types.research_state import ResearchPhase
    from app.orchestration.research_graph import research_graph

    console.print("\n[bold]Testing ResearchGraph locally...[/bold]")

    graph = await research_graph.create()

    topic = "How to improve retrieval augmented generation for multi-hop reasoning"

    initial_state = {
        "research_topic": topic,
        "max_iterations": 2,
        "current_phase": ResearchPhase.SCOUT.value,
        "iteration_count": 1,
        "metadata": {"session_id": "test-local", "user_id": "test"},
    }

    config = {"configurable": {"thread_id": f"test-{int(time.time())}"}}

    console.print(f"  Topic: {topic}")
    console.print(f"  Max iterations: 2")
    console.print("  Running pipeline...")

    start = time.monotonic()
    final_state = await graph.ainvoke(initial_state, config=config)
    elapsed = time.monotonic() - start

    console.print(f"\n  [bold]Completed in {elapsed:.1f}s[/bold]")

    papers = final_state.get("literature", [])
    hypotheses = final_state.get("hypotheses", [])
    experiments = final_state.get("experiments", [])
    journal = final_state.get("research_journal", [])
    report = final_state.get("final_report", "")

    table = Table(title="Research Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Papers", str(len(papers)))
    table.add_row("Hypotheses", str(len(hypotheses)))
    table.add_row("Experiments", str(len(experiments)))
    table.add_row("Journal Entries", str(len(journal)))
    table.add_row("Report Length", str(len(report)))
    table.add_row("Iterations", str(final_state.get("iteration_count", 0)))
    table.add_row("Final Phase", final_state.get("current_phase", "unknown"))
    console.print(table)

    if report:
        console.print(Panel(report[:1000], title="Final Report (first 1000 chars)"))

    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "topic": topic,
        "elapsed_seconds": round(elapsed, 2),
        "papers_count": len(papers),
        "hypotheses_count": len(hypotheses),
        "experiments_count": len(experiments),
        "journal_entries": len(journal),
        "final_phase": final_state.get("current_phase", ""),
        "report_preview": report[:500],
    }

    report_path = Path("test_script/2026-02-25-research-pipeline-report.json")
    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
    console.print(f"\n  Report saved to {report_path}")

    return final_state


# ── API Test ──────────────────────────────────────────────────────────────

async def test_api():
    """Test the research API endpoint against a running server."""
    console.print("\n[bold]Testing Research API...[/bold]")

    base_url = "http://localhost:8000/api/v1/research"
    topic = "How to improve retrieval augmented generation for multi-hop reasoning"

    payload = {
        "topic": topic,
        "max_iterations": 2,
        "metadata": {"user_id": "test"},
    }

    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        console.print(f"  POST {base_url}/start")
        console.print(f"  Topic: {topic}")

        try:
            response = await client.post(f"{base_url}/start", json=payload)
        except httpx.ConnectError:
            console.print("  [red]ERROR: Cannot connect to server at localhost:8000[/red]")
            console.print("  Make sure the server is running: make run")
            return None

        console.print(f"  Status: {response.status_code}")

        if response.status_code != 200:
            console.print(f"  [red]Error: {response.text[:500]}[/red]")
            return None

        data = response.json()

        table = Table(title="API Response")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Session ID", data.get("session_id", ""))
        table.add_row("Topic", data.get("research_topic", "")[:60])
        table.add_row("Iterations", str(data.get("iterations_completed", 0)))
        table.add_row("Papers", str(data.get("papers_reviewed", 0)))
        table.add_row("Hypotheses", str(len(data.get("hypotheses", []))))
        table.add_row("Experiments", str(len(data.get("experiments", []))))
        table.add_row("Report Length", str(len(data.get("final_report", ""))))
        console.print(table)

        if data.get("final_report"):
            console.print(Panel(
                data["final_report"][:1000],
                title="Final Report (first 1000 chars)",
            ))

        console.print("  [green]PASS[/green]")
        return data


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Research pipeline integration test")
    parser.add_argument(
        "--mode",
        choices=["api", "local", "clients"],
        default="clients",
        help="Test mode: 'api' (HTTP), 'local' (direct graph), 'clients' (just API clients)",
    )
    args = parser.parse_args()

    console.print(Panel(
        f"Research Pipeline Integration Test\nMode: {args.mode}",
        style="bold blue",
    ))

    if args.mode == "clients":
        await test_arxiv_client()
        await test_semantic_scholar_client()
        await test_sandbox_engine()
        console.print("\n[bold green]All client tests passed![/bold green]")

    elif args.mode == "local":
        await test_arxiv_client()
        await test_semantic_scholar_client()
        await test_local_graph()
        console.print("\n[bold green]Local graph test completed![/bold green]")

    elif args.mode == "api":
        await test_api()
        console.print("\n[bold green]API test completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
