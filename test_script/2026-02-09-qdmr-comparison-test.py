"""QDMR Pipeline Comparison Test — validates QDMR tree exploration effectiveness.

Compares QDMR-based retrieval (classify -> decompose -> tree explore -> cascade)
against baseline metrics, using real seed data in PGVector + Neo4j.

Usage:
    docker compose exec app python test_script/2026-02-09-qdmr-comparison-test.py

Or locally (requires PG + Neo4j running):
    python test_script/2026-02-09-qdmr-comparison-test.py

Outputs: JSON report with per-query comparison metrics.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.logging import logger  # noqa: E402
from app.core.types.agent_io import AgentInput, AgentStatus  # noqa: E402
from app.core.types.evidence import EvidenceGraph  # noqa: E402


# ── Test Queries ──────────────────────────────────────────────────────────
# Designed to cover different QDMR types

TEST_QUERIES = [
    {
        "query": "什么是知识图谱?",
        "expected_type": "QSelect",
        "description": "简单事实查询, 预期QSelect, 主策略RAG",
    },
    {
        "query": "知识图谱和向量数据库有什么区别?",
        "expected_type": "QComparison",
        "description": "比较查询, 预期QComparison, 主策略both",
    },
    {
        "query": "谁发明了Transformer, 这个团队后来做了什么?",
        "expected_type": "QChain",
        "description": "链式推理, 预期QChain, 主策略KG",
    },
    {
        "query": "解释知识图谱的构建方法并说明其在金融领域的应用",
        "expected_type": "QComposition",
        "description": "组合查询, 预期QComposition, 主策略both",
    },
    {
        "query": "Neo4j是否支持全文检索?",
        "expected_type": "QBoolean",
        "description": "是否判断, 预期QBoolean, 主策略RAG",
    },
    {
        "query": "列出最常用的图神经网络模型",
        "expected_type": "QAggregation",
        "description": "聚合查询, 预期QAggregation, 主策略KG",
    },
    {
        "query": "知识图谱在医疗和金融两个领域分别有哪些应用?",
        "expected_type": "QUnion",
        "description": "联合查询, 预期QUnion, 主策略both",
    },
    {
        "query": "哪些图数据库支持大规模分布式部署?",
        "expected_type": "QFilter",
        "description": "条件过滤, 预期QFilter, 主策略RAG",
    },
]


async def run_qdmr_pipeline(query: str) -> dict:
    """Run the full QDMR retrieval pipeline on a single query."""
    from app.agents.retrieval.agent import RetrievalAgent

    agent = RetrievalAgent()
    evidence_graph = EvidenceGraph()

    start_time = time.time()
    try:
        output = await agent._execute(AgentInput(
            query=query,
            evidence_graph=evidence_graph,
        ))
        elapsed = time.time() - start_time

        # Extract metrics
        metrics = output.metrics
        status = output.status.value

        # Count node types
        node_types = {}
        for n in evidence_graph.nodes.values():
            ct = n.evidence.content_type
            node_types[ct] = node_types.get(ct, 0) + 1

        # Calculate average confidence per type
        type_confidences = {}
        for n in evidence_graph.nodes.values():
            ct = n.evidence.content_type
            if ct not in type_confidences:
                type_confidences[ct] = []
            type_confidences[ct].append(n.belief.confidence)

        avg_confidences = {
            k: sum(v) / len(v) for k, v in type_confidences.items()
        }

        return {
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "total_nodes": len(evidence_graph.nodes),
            "total_edges": len(evidence_graph.edges),
            "node_type_counts": node_types,
            "avg_confidences": {k: round(v, 3) for k, v in avg_confidences.items()},
            "metrics": metrics,
            "summary": output.summary,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }


async def run_classification_only(query: str) -> dict:
    """Run only the QDMR classification step."""
    from app.agents.retrieval.query_classifier import QueryClassifier

    classifier = QueryClassifier(
        max_depth=settings.RETRIEVAL_MAX_TREE_DEPTH,
        backtrack_threshold=settings.RETRIEVAL_BACKTRACK_THRESHOLD,
    )
    evidence_graph = EvidenceGraph()

    start_time = time.time()
    try:
        result, node = await classifier.classify(query, evidence_graph)
        elapsed = time.time() - start_time

        return {
            "q_type": result.q_type.value,
            "confidence": round(result.confidence, 3),
            "reasoning": result.reasoning,
            "primary_strategy": result.strategy.primary_strategy,
            "fallback_strategy": result.strategy.fallback_strategy,
            "max_depth": result.strategy.max_decomposition_depth,
            "elapsed_seconds": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "q_type": "ERROR",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


async def main():
    """Run the comparison test suite."""
    print("=" * 70)
    print("QDMR Pipeline Comparison Test")
    print("=" * 70)

    # Phase 1: Classification accuracy test
    print("\n--- Phase 1: QDMR Classification Accuracy ---\n")
    classification_results = []
    correct = 0
    total = len(TEST_QUERIES)

    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"  [{i}/{total}] {tq['query'][:50]}...")
        result = await run_classification_only(tq["query"])
        match = result.get("q_type") == tq["expected_type"]
        if match:
            correct += 1
        symbol = "OK" if match else "MISS"
        print(f"    -> [{symbol}] classified={result.get('q_type')}, "
              f"expected={tq['expected_type']}, "
              f"conf={result.get('confidence', 0):.2f}, "
              f"time={result.get('elapsed_seconds', 0):.1f}s")
        if result.get("reasoning"):
            print(f"    -> reasoning: {result['reasoning'][:80]}")

        classification_results.append({
            "query": tq["query"],
            "expected": tq["expected_type"],
            "actual": result.get("q_type"),
            "match": match,
            "confidence": result.get("confidence", 0),
            "strategy": result.get("primary_strategy"),
            "elapsed": result.get("elapsed_seconds", 0),
        })

    accuracy = correct / total * 100
    print(f"\n  Classification Accuracy: {correct}/{total} = {accuracy:.0f}%")

    # Phase 2: Full pipeline test (subset)
    print("\n--- Phase 2: Full QDMR Pipeline Test ---\n")
    pipeline_results = []
    # Run only 3 representative queries to save time
    subset_indices = [0, 2, 3]  # QSelect, QChain, QComposition

    for idx in subset_indices:
        tq = TEST_QUERIES[idx]
        print(f"  Running: {tq['query'][:60]}...")
        result = await run_qdmr_pipeline(tq["query"])

        if result.get("error"):
            print(f"    -> ERROR: {result['error'][:100]}")
        else:
            print(f"    -> status={result['status']}, "
                  f"nodes={result['total_nodes']}, "
                  f"edges={result['total_edges']}, "
                  f"time={result['elapsed_seconds']:.1f}s")
            if result.get("metrics"):
                m = result["metrics"]
                print(f"    -> q_type={m.get('q_type')}, "
                      f"steps={m.get('tree_step_count', 0)}, "
                      f"backtracks={m.get('backtrack_count', 0)}, "
                      f"rag={m.get('breadth_result_count', 0)}, "
                      f"kg={m.get('depth_chain_count', 0)}, "
                      f"fused={m.get('fused_count', 0)}")
            if result.get("node_type_counts"):
                print(f"    -> node types: {result['node_type_counts']}")

        pipeline_results.append({
            "query": tq["query"],
            "expected_type": tq["expected_type"],
            "result": result,
        })

    # Phase 3: Summary report
    print("\n--- Summary Report ---\n")
    report = {
        "test_date": "2026-02-09",
        "classification_accuracy": f"{accuracy:.0f}%",
        "classification_results": classification_results,
        "pipeline_results": pipeline_results,
        "config": {
            "backtrack_threshold": settings.RETRIEVAL_BACKTRACK_THRESHOLD,
            "max_branch_retries": settings.RETRIEVAL_MAX_BRANCH_RETRIES,
            "max_tree_depth": settings.RETRIEVAL_MAX_TREE_DEPTH,
            "top_k": settings.RETRIEVAL_TOP_K,
            "cascade_rounds": settings.RETRIEVAL_CASCADE_ROUNDS,
        },
    }

    # Key findings
    avg_class_time = sum(r["elapsed"] for r in classification_results) / len(classification_results)
    print(f"  Classification accuracy:     {accuracy:.0f}%")
    print(f"  Avg classification time:     {avg_class_time:.1f}s")

    successful_pipelines = [r for r in pipeline_results if r["result"].get("status") != "error"]
    if successful_pipelines:
        avg_nodes = sum(r["result"]["total_nodes"] for r in successful_pipelines) / len(successful_pipelines)
        avg_time = sum(r["result"]["elapsed_seconds"] for r in successful_pipelines) / len(successful_pipelines)
        total_backtracks = sum(r["result"].get("metrics", {}).get("backtrack_count", 0) for r in successful_pipelines)
        print(f"  Avg pipeline nodes:          {avg_nodes:.0f}")
        print(f"  Avg pipeline time:           {avg_time:.1f}s")
        print(f"  Total backtracks triggered:  {total_backtracks}")

    # Save report
    report_path = Path(__file__).parent / "2026-02-09-qdmr-comparison-report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Full report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
