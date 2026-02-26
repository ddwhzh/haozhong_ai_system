"""Dataset accuracy evaluation for HaoZhong multi-agent pipeline.

Runs ~10 benchmark queries against `/api/v1/agent/chat` and evaluates
answer accuracy with keyword-group matching + fallback detection.

Usage:
    python test_script/2026-02-10-dataset-accuracy-eval.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import httpx


API_URL = "http://localhost:8000/api/v1/agent/chat"
REPORT_PATH = Path("test_script/2026-02-10-dataset-accuracy-report.json")


@dataclass
class EvalCase:
    query: str
    # Each group means one fact point. If any keyword in the group appears, that fact is hit.
    keyword_groups: List[List[str]]
    min_group_hits: int
    note: str = ""


@dataclass
class EvalResult:
    query: str
    fallback: bool
    group_hits: int
    required_hits: int
    matched_keywords: List[str]
    is_correct: bool
    latency_seconds: float
    phase_completed: str
    response_preview: str
    note: str = ""


DATASET: List[EvalCase] = [
    EvalCase(
        query="什么是知识图谱?",
        keyword_groups=[
            ["图结构", "graph", "图模型", "语义网络"],
            ["实体", "entity", "节点"],
            ["关系", "relation", "边", "链接"],
        ],
        min_group_hits=2,
        note="定义类基础问题",
    ),
    EvalCase(
        query="知识图谱由哪些核心组件构成?",
        keyword_groups=[
            ["实体", "entity", "节点"],
            ["关系", "relation", "边"],
            ["属性", "attribute", "property"],
            ["三元组", "triple", "triplet"],
        ],
        min_group_hits=2,
        note="组件枚举",
    ),
    EvalCase(
        query="Neo4j使用什么查询语言?",
        keyword_groups=[["cypher"]],
        min_group_hits=1,
        note="单事实检索",
    ),
    EvalCase(
        query="RAG的典型流程是什么?",
        keyword_groups=[
            ["query", "查询", "提问", "问题"],
            ["embedding", "嵌入", "向量化"],
            ["vector search", "向量检索", "向量数据库", "向量搜索", "相似性搜索"],
            ["context", "上下文", "拼接", "组装"],
            ["generation", "生成", "回答", "输出"],
        ],
        min_group_hits=3,
        note="流程型问题",
    ),
    EvalCase(
        query="高级RAG优化策略有哪些?",
        keyword_groups=[
            ["query分解", "查询分解", "查询重写", "query rewriting"],
            ["混合检索", "hybrid", "多路检索", "多源检索"],
            ["重排序", "reranking", "rerank", "cross-encoder"],
            ["graph rag", "图增强", "知识图谱增强"],
            ["分块", "chunking", "chunk", "切分"],
            ["嵌入", "embedding", "微调", "fine-tun"],
            ["提示", "prompt", "提示工程", "prompt engineering"],
        ],
        min_group_hits=2,
        note="多点列举",
    ),
    EvalCase(
        query="Fast R-CNN的核心思想是什么?",
        keyword_groups=[["两阶段", "two-stage"], ["proposal", "候选框", "roi"]],
        min_group_hits=1,
        note="模型概念",
    ),
    EvalCase(
        query="DETR借助什么算法进行最优匹配?",
        keyword_groups=[
            ["匈牙利算法", "hungarian", "bipartite matching", "bipartite", "二分匹配", "二部图"],
        ],
        min_group_hits=1,
        note="算法问答",
    ),
    EvalCase(
        query="LangGraph的核心概念有哪些?",
        keyword_groups=[
            ["state", "状态", "状态化", "stateful"],
            ["node", "节点", "node节点"],
            ["edge", "边", "条件边", "conditional edge"],
            ["checkpoint", "检查点", "持久化", "持久性", "persistence"],
            ["command", "命令", "路由", "dispatch"],
        ],
        min_group_hits=3,
        note="框架概念列表",
    ),
    EvalCase(
        query="多Agent系统有哪些优势?",
        keyword_groups=[
            ["模块化", "分工", "专业化", "独立"],
            ["可扩展", "扩展性", "扩展"],
            ["鲁棒", "容错", "冗余", "可靠"],
            ["协作", "协同", "协调"],
            ["并行", "并发", "效率"],
        ],
        min_group_hits=2,
        note="优点列举",
    ),
    EvalCase(
        query="知识图谱在医疗和金融领域分别有哪些应用?",
        keyword_groups=[
            ["医疗", "医学", "临床"],
            ["金融", "银行", "保险"],
            ["诊断", "辅助诊断", "临床诊断"],
            ["风控", "反欺诈", "风险"],
        ],
        min_group_hits=2,
        note="跨领域应用",
    ),
]


def _normalize(text: str) -> str:
    return (text or "").lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k.lower() in text for k in keywords)


def _is_fallback_answer(answer: str) -> bool:
    lower = _normalize(answer)
    fallback_signals = [
        "缺少与你问题直接相关的证据",
        "暂时无法给出可靠结论",
        "insufficient evidence",
        "lacks directly relevant evidence",
    ]
    return any(s in lower for s in fallback_signals)


async def evaluate_case(client: httpx.AsyncClient, case: EvalCase) -> EvalResult:
    start = time.time()
    resp = await client.post(
        API_URL,
        json={
            "query": case.query,
            "max_iterations": 10,
            "metadata": {"eval": "dataset_accuracy_2026_02_10"},
        },
        timeout=240.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    answer = payload.get("response", "") or ""
    phase_completed = payload.get("phase_completed", "")
    latency = round(time.time() - start, 2)

    normalized = _normalize(answer)
    fallback = _is_fallback_answer(answer)

    hit_count = 0
    matched_keywords: List[str] = []
    for group in case.keyword_groups:
        if _contains_any(normalized, [g.lower() for g in group]):
            hit_count += 1
            matched_keywords.append(group[0])

    is_correct = (not fallback) and (hit_count >= case.min_group_hits)
    preview = answer[:360].replace("\n", " ")

    return EvalResult(
        query=case.query,
        fallback=fallback,
        group_hits=hit_count,
        required_hits=case.min_group_hits,
        matched_keywords=matched_keywords,
        is_correct=is_correct,
        latency_seconds=latency,
        phase_completed=phase_completed,
        response_preview=preview,
        note=case.note,
    )


async def main() -> None:
    results: List[EvalResult] = []
    async with httpx.AsyncClient() as client:
        for idx, case in enumerate(DATASET, start=1):
            case_start = time.time()
            try:
                result = await evaluate_case(client, case)
                results.append(result)
                print(
                    f"[{idx:02d}] {'PASS' if result.is_correct else 'FAIL'} | "
                    f"hits={result.group_hits}/{result.required_hits} | "
                    f"{result.latency_seconds}s | {case.query}"
                )
            except Exception as e:
                fail_result = EvalResult(
                    query=case.query,
                    fallback=False,
                    group_hits=0,
                    required_hits=case.min_group_hits,
                    matched_keywords=[],
                    is_correct=False,
                    latency_seconds=round(time.time() - case_start, 2),
                    phase_completed="error",
                    response_preview=f"ERROR: {type(e).__name__}: {e}",
                    note=case.note,
                )
                results.append(fail_result)
                print(f"[{idx:02d}] ERROR | {case.query} | {type(e).__name__}: {e}")

    total = len(results)
    passed = sum(1 for r in results if r.is_correct)
    accuracy = round((passed / total) * 100.0, 2) if total else 0.0
    avg_latency = round(sum(r.latency_seconds for r in results) / total, 2) if total else 0.0
    fallback_count = sum(1 for r in results if r.fallback)

    summary = {
        "dataset_size": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy_percent": accuracy,
        "avg_latency_seconds": avg_latency,
        "fallback_count": fallback_count,
        "timestamp_unix": int(time.time()),
    }

    report = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
