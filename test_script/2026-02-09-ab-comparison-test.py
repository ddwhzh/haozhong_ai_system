"""A/B Comparison Test: QDMR Pipeline vs Simple Pipeline.

Quantitatively compares:
  Mode A (Simple): Direct RAG + KG search on original query, then cascade fuse.
  Mode B (QDMR):   Classify -> Decompose -> Tree Explore -> Cascade Fuse.

Metrics:
  - Retrieval coverage: number of unique evidence nodes
  - Average confidence of retrieval results
  - Fused evidence quality (confidence, completeness)
  - Backtracking effectiveness (QDMR only)
  - Latency (seconds)

Usage:
    python test_script/2026-02-09-ab-comparison-test.py
"""

import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.logging import logger  # noqa: E402
from app.core.types.agent_io import AgentInput, AgentStatus  # noqa: E402
from app.core.types.belief import Belief  # noqa: E402
from app.core.types.evidence import (  # noqa: E402
    Evidence,
    EvidenceGraph,
    EvidenceNode,
    EdgeRelation,
)
from app.core.types.intent import Intent, IntentType  # noqa: E402


# ── Test Queries ─────────────────────────────────────────────────────────
# Categorized by expected complexity to show where QDMR should help.

TEST_QUERIES = [
    # Simple queries: QDMR overhead may not help
    {
        "query": "什么是知识图谱?",
        "expected_type": "QSelect",
        "complexity": "simple",
        "description": "简单事实查询",
    },
    {
        "query": "Neo4j是否支持全文检索?",
        "expected_type": "QBoolean",
        "complexity": "simple",
        "description": "是否判断查询",
    },
    # Medium queries: QDMR should help with decomposition
    {
        "query": "知识图谱和向量数据库有什么区别?",
        "expected_type": "QComparison",
        "complexity": "medium",
        "description": "比较查询 — 需要两个检索目标",
    },
    {
        "query": "哪些图数据库支持大规模分布式部署?",
        "expected_type": "QFilter",
        "complexity": "medium",
        "description": "条件过滤查询",
    },
    # Complex queries: QDMR should significantly help
    {
        "query": "谁发明了Transformer, 这个团队后来做了什么?",
        "expected_type": "QChain",
        "complexity": "complex",
        "description": "链式推理 — 多跳依赖",
    },
    {
        "query": "解释知识图谱的构建方法并说明其在金融领域的应用",
        "expected_type": "QComposition",
        "complexity": "complex",
        "description": "组合查询 — 多子任务",
    },
    {
        "query": "知识图谱在医疗和金融两个领域分别有哪些应用?",
        "expected_type": "QUnion",
        "complexity": "complex",
        "description": "联合查询 — 多维度",
    },
]


@dataclass
class ModeResult:
    """Result from a single mode run."""
    mode: str
    query: str
    success: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    total_nodes: int = 0
    total_edges: int = 0
    rag_nodes: int = 0
    kg_nodes: int = 0
    reasoning_chains: int = 0
    fused_nodes: int = 0
    avg_retrieval_confidence: float = 0.0
    fused_confidence: float = 0.0
    backtrack_count: int = 0
    tree_step_count: int = 0
    node_types: Dict[str, int] = field(default_factory=dict)


def _analyze_graph(graph: EvidenceGraph) -> Dict[str, Any]:
    """Analyze evidence graph metrics."""
    node_types = {}
    confidences_by_type = {}
    for n in graph.nodes.values():
        ct = n.evidence.content_type
        node_types[ct] = node_types.get(ct, 0) + 1
        if ct not in confidences_by_type:
            confidences_by_type[ct] = []
        confidences_by_type[ct].append(n.belief.confidence)

    retrieval_types = ("document_chunk", "kg_match", "reasoning_chain")
    retrieval_confs = []
    for ct in retrieval_types:
        retrieval_confs.extend(confidences_by_type.get(ct, []))

    fused_confs = confidences_by_type.get("fused_evidence", [])

    return {
        "node_types": node_types,
        "rag_nodes": node_types.get("document_chunk", 0),
        "kg_nodes": node_types.get("kg_match", 0),
        "reasoning_chains": node_types.get("reasoning_chain", 0),
        "fused_nodes": node_types.get("fused_evidence", 0),
        "tree_steps": node_types.get("tree_step", 0),
        "avg_retrieval_confidence": (
            sum(retrieval_confs) / len(retrieval_confs) if retrieval_confs else 0.0
        ),
        "fused_confidence": (
            fused_confs[-1] if fused_confs else 0.0
        ),
    }


# ── Mode A: Simple Pipeline ─────────────────────────────────────────────

async def run_simple_pipeline(query: str) -> ModeResult:
    """Simple mode: direct RAG + KG search on original query, then cascade fuse.

    No classification, no decomposition, no tree exploration.
    Just: create a sub_query node -> VectorRetriever + KGRetriever -> CascadeFuser.
    """
    from app.agents.retrieval.cascade_fuser import CascadeFuser
    from app.agents.retrieval.kg_retriever import KGRetriever
    from app.agents.retrieval.vector_retriever import VectorRetriever

    result = ModeResult(mode="simple", query=query)
    evidence_graph = EvidenceGraph()

    start = time.time()
    try:
        # Create a single sub_query node (simulating no decomposition)
        root_node = evidence_graph.add_evidence(
            evidence=Evidence(
                content=query,
                content_type="sub_query",
                source="simple_baseline",
            ),
            belief=Belief(
                content=f"Original query: {query}",
                confidence=1.0,
                source_agent="baseline",
            ),
            intent=Intent(
                intent_type=IntentType.RETRIEVE,
                description="Direct query retrieval (no decomposition)",
                source_agent="baseline",
            ),
        )

        # Direct RAG search
        vector_retriever = VectorRetriever(top_k=settings.RETRIEVAL_TOP_K)
        rag_nodes = await vector_retriever.retrieve([root_node], evidence_graph)

        # Direct KG search
        kg_retriever = KGRetriever(max_hops=settings.RETRIEVAL_KG_MAX_HOPS)
        kg_nodes = await kg_retriever.retrieve([root_node], evidence_graph)

        # Separate by type
        breadth = [n for n in rag_nodes if n.evidence.content_type == "document_chunk"]
        depth = [
            n for n in kg_nodes
            if n.evidence.content_type in ("kg_match", "reasoning_chain")
        ]

        # Cascade fuse
        fuser = CascadeFuser(max_rounds=settings.RETRIEVAL_CASCADE_ROUNDS)
        fused = await fuser.fuse(
            query=query,
            evidence_graph=evidence_graph,
            breadth_nodes=breadth,
            depth_nodes=depth,
        )

        elapsed = time.time() - start
        analysis = _analyze_graph(evidence_graph)

        result.success = True
        result.elapsed_seconds = round(elapsed, 2)
        result.total_nodes = len(evidence_graph.nodes)
        result.total_edges = len(evidence_graph.edges)
        result.rag_nodes = analysis["rag_nodes"]
        result.kg_nodes = analysis["kg_nodes"]
        result.reasoning_chains = analysis["reasoning_chains"]
        result.fused_nodes = analysis["fused_nodes"]
        result.avg_retrieval_confidence = round(analysis["avg_retrieval_confidence"], 3)
        result.fused_confidence = round(analysis["fused_confidence"], 3)
        result.node_types = analysis["node_types"]

    except Exception as e:
        result.elapsed_seconds = round(time.time() - start, 2)
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ── Mode B: QDMR Pipeline ───────────────────────────────────────────────

async def run_qdmr_pipeline(query: str) -> ModeResult:
    """QDMR mode: classify -> decompose -> tree explore -> cascade fuse."""
    from app.agents.retrieval.agent import RetrievalAgent

    result = ModeResult(mode="qdmr", query=query)
    evidence_graph = EvidenceGraph()

    start = time.time()
    try:
        agent = RetrievalAgent()
        output = await agent._execute(AgentInput(
            query=query,
            evidence_graph=evidence_graph,
        ))

        elapsed = time.time() - start
        analysis = _analyze_graph(evidence_graph)

        result.success = True
        result.elapsed_seconds = round(elapsed, 2)
        result.total_nodes = len(evidence_graph.nodes)
        result.total_edges = len(evidence_graph.edges)
        result.rag_nodes = analysis["rag_nodes"]
        result.kg_nodes = analysis["kg_nodes"]
        result.reasoning_chains = analysis["reasoning_chains"]
        result.fused_nodes = analysis["fused_nodes"]
        result.avg_retrieval_confidence = round(analysis["avg_retrieval_confidence"], 3)
        result.fused_confidence = round(analysis["fused_confidence"], 3)
        result.tree_step_count = analysis["tree_steps"]
        result.backtrack_count = int(output.metrics.get("backtrack_count", 0))
        result.node_types = analysis["node_types"]

    except Exception as e:
        result.elapsed_seconds = round(time.time() - start, 2)
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ── Main ─────────────────────────────────────────────────────────────────

def _print_comparison(query_info: dict, simple: ModeResult, qdmr: ModeResult):
    """Pretty-print A/B comparison for one query."""
    q = query_info["query"]
    cx = query_info["complexity"]
    exp = query_info["expected_type"]

    print(f"\n  Query: {q}")
    print(f"  Complexity: {cx} | Expected: {exp}")
    print(f"  {'Metric':<28} {'Simple':>10} {'QDMR':>10} {'Delta':>10}")
    print(f"  {'─' * 60}")

    def row(label, s_val, q_val, fmt=".2f", better="higher"):
        if isinstance(s_val, float):
            s_str = f"{s_val:{fmt}}"
            q_str = f"{q_val:{fmt}}"
            delta = q_val - s_val
            d_str = f"{delta:+{fmt}}"
            if better == "higher":
                color = "+" if delta > 0 else ("-" if delta < 0 else "=")
            else:
                color = "+" if delta < 0 else ("-" if delta > 0 else "=")
        else:
            s_str = str(s_val)
            q_str = str(q_val)
            d_str = str(q_val - s_val) if isinstance(s_val, (int, float)) else "—"
            color = ""
        marker = {"+" : " ▲", "-": " ▼", "=": " =", "": ""}[color]
        print(f"  {label:<28} {s_str:>10} {q_str:>10} {d_str:>10}{marker}")

    s_ok = "OK" if simple.success else "FAIL"
    q_ok = "OK" if qdmr.success else "FAIL"
    print(f"  {'Status':<28} {s_ok:>10} {q_ok:>10}")

    if simple.success and qdmr.success:
        row("RAG nodes", simple.rag_nodes, qdmr.rag_nodes, "d")
        row("KG nodes", simple.kg_nodes, qdmr.kg_nodes, "d")
        row("Reasoning chains", simple.reasoning_chains, qdmr.reasoning_chains, "d")
        row("Fused nodes", simple.fused_nodes, qdmr.fused_nodes, "d")
        row("Total evidence", simple.total_nodes, qdmr.total_nodes, "d")
        row("Avg retrieval conf.", simple.avg_retrieval_confidence, qdmr.avg_retrieval_confidence)
        row("Fused confidence", simple.fused_confidence, qdmr.fused_confidence)
        row("QDMR tree steps", 0, qdmr.tree_step_count, "d")
        row("Backtracks", 0, qdmr.backtrack_count, "d")
        row("Latency (s)", simple.elapsed_seconds, qdmr.elapsed_seconds, ".1f", "lower")
    elif simple.error:
        print(f"  Simple error: {simple.error[:80]}")
    elif qdmr.error:
        print(f"  QDMR error: {qdmr.error[:80]}")


async def main():
    print("=" * 72)
    print("  A/B Comparison: Simple Pipeline vs QDMR Pipeline")
    print("  (after bug fixes: content_type match + metrics type)")
    print("=" * 72)

    all_results = []

    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 72}")
        print(f"  [{i}/{len(TEST_QUERIES)}] {tq['description']}: {tq['query']}")
        print(f"{'─' * 72}")

        # Run both modes
        print(f"  Running Simple mode...")
        simple = await run_simple_pipeline(tq["query"])
        print(f"    -> {simple.elapsed_seconds}s, nodes={simple.total_nodes}, "
              f"rag={simple.rag_nodes}, kg={simple.kg_nodes}")

        print(f"  Running QDMR mode...")
        qdmr = await run_qdmr_pipeline(tq["query"])
        print(f"    -> {qdmr.elapsed_seconds}s, nodes={qdmr.total_nodes}, "
              f"rag={qdmr.rag_nodes}, kg={qdmr.kg_nodes}, "
              f"steps={qdmr.tree_step_count}, backtracks={qdmr.backtrack_count}")

        _print_comparison(tq, simple, qdmr)

        all_results.append({
            "query": tq["query"],
            "complexity": tq["complexity"],
            "expected_type": tq["expected_type"],
            "simple": asdict(simple),
            "qdmr": asdict(qdmr),
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  AGGREGATE SUMMARY")
    print(f"{'=' * 72}")

    # Filter successful pairs
    pairs = [
        r for r in all_results
        if r["simple"]["success"] and r["qdmr"]["success"]
    ]

    if not pairs:
        print("  No successful pairs to compare!")
    else:
        # Group by complexity
        for cx in ["simple", "medium", "complex"]:
            cx_pairs = [p for p in pairs if p["complexity"] == cx]
            if not cx_pairs:
                continue

            print(f"\n  [{cx.upper()} queries] ({len(cx_pairs)} queries)")

            def avg(key, mode):
                vals = [p[mode][key] for p in cx_pairs]
                return sum(vals) / len(vals) if vals else 0

            s_rag = avg("rag_nodes", "simple")
            q_rag = avg("rag_nodes", "qdmr")
            s_kg = avg("kg_nodes", "simple")
            q_kg = avg("kg_nodes", "qdmr")
            s_rc = avg("reasoning_chains", "simple")
            q_rc = avg("reasoning_chains", "qdmr")
            s_conf = avg("avg_retrieval_confidence", "simple")
            q_conf = avg("avg_retrieval_confidence", "qdmr")
            s_fused = avg("fused_confidence", "simple")
            q_fused = avg("fused_confidence", "qdmr")
            s_time = avg("elapsed_seconds", "simple")
            q_time = avg("elapsed_seconds", "qdmr")
            q_bt = avg("backtrack_count", "qdmr")

            print(f"    {'Metric':<28} {'Simple':>10} {'QDMR':>10}")
            print(f"    {'─' * 50}")
            print(f"    {'Avg RAG nodes':<28} {s_rag:>10.1f} {q_rag:>10.1f}")
            print(f"    {'Avg KG nodes':<28} {s_kg:>10.1f} {q_kg:>10.1f}")
            print(f"    {'Avg reasoning chains':<28} {s_rc:>10.1f} {q_rc:>10.1f}")
            print(f"    {'Avg retrieval confidence':<28} {s_conf:>10.3f} {q_conf:>10.3f}")
            print(f"    {'Avg fused confidence':<28} {s_fused:>10.3f} {q_fused:>10.3f}")
            print(f"    {'Avg latency (s)':<28} {s_time:>10.1f} {q_time:>10.1f}")
            print(f"    {'Avg backtracks':<28} {'—':>10} {q_bt:>10.1f}")

        # Overall
        print(f"\n  [OVERALL] ({len(pairs)} queries)")
        s_total_nodes = sum(p["simple"]["total_nodes"] for p in pairs)
        q_total_nodes = sum(p["qdmr"]["total_nodes"] for p in pairs)
        s_avg_conf = sum(p["simple"]["avg_retrieval_confidence"] for p in pairs) / len(pairs)
        q_avg_conf = sum(p["qdmr"]["avg_retrieval_confidence"] for p in pairs) / len(pairs)
        s_avg_fused = sum(p["simple"]["fused_confidence"] for p in pairs) / len(pairs)
        q_avg_fused = sum(p["qdmr"]["fused_confidence"] for p in pairs) / len(pairs)
        s_avg_time = sum(p["simple"]["elapsed_seconds"] for p in pairs) / len(pairs)
        q_avg_time = sum(p["qdmr"]["elapsed_seconds"] for p in pairs) / len(pairs)

        conf_improvement = ((q_avg_conf - s_avg_conf) / s_avg_conf * 100) if s_avg_conf else 0
        fused_improvement = ((q_avg_fused - s_avg_fused) / s_avg_fused * 100) if s_avg_fused else 0
        latency_overhead = ((q_avg_time - s_avg_time) / s_avg_time * 100) if s_avg_time else 0

        print(f"    Total evidence nodes:    Simple={s_total_nodes}, QDMR={q_total_nodes}")
        print(f"    Avg retrieval conf:      Simple={s_avg_conf:.3f}, QDMR={q_avg_conf:.3f} ({conf_improvement:+.1f}%)")
        print(f"    Avg fused conf:          Simple={s_avg_fused:.3f}, QDMR={q_avg_fused:.3f} ({fused_improvement:+.1f}%)")
        print(f"    Avg latency:             Simple={s_avg_time:.1f}s, QDMR={q_avg_time:.1f}s ({latency_overhead:+.1f}%)")

        # Verdict
        print(f"\n  {'─' * 50}")
        if q_avg_fused > s_avg_fused and q_avg_conf >= s_avg_conf:
            print("  VERDICT: QDMR IMPROVES quality (higher fused confidence & retrieval confidence)")
            if latency_overhead > 100:
                print(f"  WARNING: Latency overhead is {latency_overhead:.0f}% — consider caching classification")
        elif q_avg_fused >= s_avg_fused:
            print("  VERDICT: QDMR maintains quality with additional structure")
        else:
            print("  VERDICT: QDMR does NOT improve quality — needs investigation")

    # Save full report
    report = {
        "test_date": "2026-02-09",
        "description": "A/B comparison: Simple vs QDMR retrieval pipeline",
        "bug_fixes_applied": [
            "metrics Dict[str, float] -> Dict[str, Any] for q_type string",
            "VectorRetriever/KGRetriever content_type filter: accept 'tree_step' alongside 'sub_query'",
        ],
        "config": {
            "backtrack_threshold": settings.RETRIEVAL_BACKTRACK_THRESHOLD,
            "max_branch_retries": settings.RETRIEVAL_MAX_BRANCH_RETRIES,
            "max_tree_depth": settings.RETRIEVAL_MAX_TREE_DEPTH,
            "top_k": settings.RETRIEVAL_TOP_K,
            "cascade_rounds": settings.RETRIEVAL_CASCADE_ROUNDS,
            "kg_max_hops": settings.RETRIEVAL_KG_MAX_HOPS,
        },
        "results": all_results,
    }

    report_path = Path(__file__).parent / "2026-02-09-ab-comparison-report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Full report: {report_path}")
    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    asyncio.run(main())
