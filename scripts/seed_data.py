"""Seed data loader: populates PGVector + Neo4j with AI/KG demo dataset.

Usage:
    python scripts/seed_data.py

Connects to:
    - PostgreSQL/PGVector at localhost:5432
    - Neo4j at bolt://localhost:7687

Uses GLM embedding-3 model via OpenAI-compatible API for embeddings.
"""

import asyncio
import json
import os
import sys
import uuid

import httpx
import psycopg
from neo4j import AsyncGraphDatabase

# ── Configuration ────────────────────────────────────────────────────────

PG_DSN = "postgresql://postgres:postgres@localhost:5432/haozhong_db"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "neo4jpassword")
COLLECTION = "documents"
EMBEDDING_DIM = 2048
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")
EMBEDDING_MODEL = "embedding-3"

# ── Demo Documents (Chinese, AI/KG domain) ───────────────────────────────

DOCUMENTS = [
    {
        "id": "doc-kg-def",
        "content": "知识图谱(Knowledge Graph)是一种用图结构来组织和表示知识的技术。它由实体(节点)和关系(边)组成,能够表达实体之间的语义关联。Google于2012年首次提出知识图谱概念,用于增强搜索引擎的语义理解能力。知识图谱的核心思想是将现实世界中的事物及其相互关系映射为图结构,使计算机能够理解和推理。",
        "metadata": {"topic": "knowledge_graph", "aspect": "definition", "source": "encyclopedia"}
    },
    {
        "id": "doc-kg-components",
        "content": "知识图谱由三个核心组件构成:实体(Entity)、关系(Relation)和属性(Property)。实体是现实世界中可区分的事物,如人、地点、概念;关系描述实体之间的语义连接,如'属于'、'创建了';属性是实体的特征描述,如'成立日期'、'创始人'。这三者共同构成了知识的三元组(Triple)表示:<主体, 谓词, 客体>。",
        "metadata": {"topic": "knowledge_graph", "aspect": "components", "source": "textbook"}
    },
    {
        "id": "doc-kg-construction",
        "content": "知识图谱的构建主要包括四个步骤:1)知识抽取:从非结构化文本中提取实体、关系和属性,常用方法包括命名实体识别(NER)、关系抽取(RE)和事件抽取;2)知识融合:将来自不同数据源的知识进行对齐和合并,解决实体消歧和共指消解问题;3)知识存储:选择合适的存储方案,如图数据库Neo4j或三元组存储;4)知识推理:基于已有知识推断新知识,常用方法包括基于规则的推理和基于表示学习的推理。",
        "metadata": {"topic": "knowledge_graph", "aspect": "construction", "source": "survey_paper"}
    },
    {
        "id": "doc-kg-application",
        "content": "知识图谱在多个领域有广泛应用:1)智能搜索:Google知识面板(Knowledge Panel)利用知识图谱提供结构化搜索结果;2)智能问答:基于知识图谱的问答系统可以回答复杂的多跳推理问题;3)推荐系统:利用知识图谱中的实体关系进行基于语义的推荐;4)金融风控:构建企业关系图谱用于反欺诈和风险传导分析;5)医疗健康:构建疾病-症状-药物知识图谱辅助临床诊断。",
        "metadata": {"topic": "knowledge_graph", "aspect": "applications", "source": "industry_report"}
    },
    {
        "id": "doc-kg-neo4j",
        "content": "Neo4j是目前最流行的原生图数据库,专为存储和查询图结构数据设计。它使用Cypher查询语言,支持ACID事务、高可用集群部署。Neo4j的优势在于:1)原生图存储引擎,遍历性能极高;2)Cypher语言直观易用,适合表达图模式匹配;3)支持BFS/DFS等图算法原语;4)丰富的生态系统,包括可视化工具Neo4j Browser和图数据科学库GDS。对于知识图谱场景,Neo4j是首选的存储方案之一。",
        "metadata": {"topic": "neo4j", "aspect": "overview", "source": "documentation"}
    },
    {
        "id": "doc-rag-overview",
        "content": "检索增强生成(RAG, Retrieval-Augmented Generation)是一种结合信息检索和文本生成的技术范式。其核心思想是:在LLM生成回答之前,先从外部知识库中检索相关文档,将检索结果作为上下文提供给LLM,从而减少幻觉(Hallucination)并提高回答的准确性和时效性。RAG的典型流程:Query -> Embedding -> Vector Search -> Context Assembly -> LLM Generation。",
        "metadata": {"topic": "rag", "aspect": "overview", "source": "survey_paper"}
    },
    {
        "id": "doc-rag-advanced",
        "content": "高级RAG技术包括多种优化策略:1)Query分解:将复杂查询拆解为多个子查询,分别检索后融合结果,提升召回率;2)混合检索:结合向量检索(语义)和稀疏检索(关键词)的优势;3)重排序(Reranking):对初检结果进行二次排序,提高精确度;4)自适应RAG:根据查询复杂度动态选择检索策略;5)Graph RAG:结合知识图谱进行多跳推理,解决需要深度推理的复杂问题。",
        "metadata": {"topic": "rag", "aspect": "advanced", "source": "research_paper"}
    },
    {
        "id": "doc-embedding",
        "content": "向量嵌入(Embedding)是将文本、图像等非结构化数据转换为高维向量的技术。在知识图谱和RAG系统中,embedding扮演关键角色:1)文本embedding用于语义相似度搜索,如OpenAI的text-embedding-3-small和智谱的embedding-3;2)知识图谱embedding(如TransE、RotatE)用于链接预测和知识推理;3)跨模态embedding用于多模态检索。好的embedding应该满足:语义相近的内容在向量空间中距离近,语义不同的内容距离远。",
        "metadata": {"topic": "embedding", "aspect": "overview", "source": "textbook"}
    },
    {
        "id": "doc-multi-agent",
        "content": "多Agent系统(Multi-Agent System, MAS)是由多个自主Agent协同工作的系统。在AI应用中,多Agent架构的优势包括:1)模块化:每个Agent专注一个子任务(如检索、生成、评估),降低复杂度;2)可扩展:新增Agent不影响已有Agent;3)鲁棒性:单个Agent失败不会导致整体失败;4)可优化:各Agent可独立调优。典型的多Agent编排框架包括LangGraph、AutoGen和CrewAI。",
        "metadata": {"topic": "multi_agent", "aspect": "overview", "source": "survey_paper"}
    },
    {
        "id": "doc-langgraph",
        "content": "LangGraph是LangChain团队开发的Agent编排框架,基于有向图(DAG)进行状态管理和流程控制。核心概念:1)State:全局共享状态,所有节点可读写;2)Node:图中的处理节点,可以是Agent、工具或函数;3)Edge:节点之间的连接,支持条件路由;4)Checkpoint:状态快照,支持断点恢复和Human-In-The-Loop;5)Command:路由指令,只有Gateway/Orchestrator可以发出。LangGraph适合构建需要复杂流程控制的多Agent系统。",
        "metadata": {"topic": "langgraph", "aspect": "overview", "source": "documentation"}
    },
    {
        "id": "doc-fast-rcnn",
        "content": "Fast R-CNN是Ross Girshick于2015年提出的目标检测算法。其核心思想是两阶段检测:Stage 1(Region Proposal Network, RPN):生成候选区域(Proposals);Stage 2(ROI Head):对每个候选区域进行分类和回归。这种'先提议后分类'的范式可以迁移到NLP和Agent系统中:先将复杂任务分解为多个同结构的子任务(Proposals),再对每个子任务从已有证据中提取信息(类似ROI Pooling从特征图中提取特征)。",
        "metadata": {"topic": "fast_rcnn", "aspect": "concept", "source": "research_paper"}
    },
    {
        "id": "doc-detr",
        "content": "DETR(Detection Transformer)是Facebook于2020年提出的端到端目标检测模型。其核心创新是使用匈牙利算法(Hungarian Algorithm)进行预测与目标的最优匹配。匈牙利算法解决二部图最优匹配问题:给定N个预测和M个目标,找到总代价最小的匹配方案。在评估系统中,可以借鉴这一思想:将生成内容视为'预测',将证据源视为'目标',通过匈牙利匹配评估生成内容与证据的对齐程度,实现自动化质量评估。",
        "metadata": {"topic": "detr", "aspect": "hungarian_matching", "source": "research_paper"}
    },
    {
        "id": "doc-evaluation-scoring",
        "content": "传统的向量相似度评估存在两个问题:1)对称性问题:cos(A,B)=cos(B,A),但在评估场景中,'A是否忠实于B'和'B是否忠实于A'是不同的;2)细粒度区分度差:高维embedding空间中,大多数向量的余弦相似度集中在较窄区间。解决方案包括:自监督去噪(去除公共噪声方向,保留区分性信息)和引入稀疏矩阵分量(类BM25的非对称词项重叠得分)。组合公式为:score = alpha * dense_denoised + (1-alpha) * sparse。",
        "metadata": {"topic": "evaluation", "aspect": "scoring", "source": "research_paper"}
    },
    {
        "id": "doc-hitl",
        "content": "Human-In-The-Loop(HITL)是一种将人工审核嵌入AI系统决策流程的模式。在多Agent管道中,HITL通常在以下环节介入:1)质量边界区间:当评估分数处于'不确定'区间时请求人工确认;2)高风险决策:涉及敏感内容或重要决策时;3)模型纠偏:人工反馈用于改进Agent行为。LangGraph原生支持HITL,通过interrupt_before机制在指定节点前暂停管道,等待人工输入后恢复执行。",
        "metadata": {"topic": "hitl", "aspect": "overview", "source": "documentation"}
    },
    {
        "id": "doc-prompt-optimization",
        "content": "自动Prompt优化是通过程序化方法迭代改进LLM的Prompt,以提升输出质量。在评估Agent中,当生成质量低于阈值时触发Prompt优化:1)诊断弱环节:识别低分section的根因(幻觉/缺上下文/粒度错误);2)生成Prompt补丁:针对system prompt、proposal模板、extraction模板分别生成修改建议;3)评估改善:预测每个补丁的期望改善幅度;4)增量应用:选择最有价值的补丁应用到下一轮生成。这构成了一个自动化的Prompt改进闭环。",
        "metadata": {"topic": "prompt_optimization", "aspect": "auto_prompt", "source": "research_paper"}
    },
]

# ── Knowledge Graph Entities & Relations ─────────────────────────────────

KG_ENTITIES = [
    ("e-kg", "Concept", "知识图谱", "一种用图结构组织和表示知识的技术,由实体和关系组成"),
    ("e-entity", "Concept", "实体", "知识图谱中的节点,表示现实世界中可区分的事物"),
    ("e-relation", "Concept", "关系", "知识图谱中的边,描述实体之间的语义连接"),
    ("e-triple", "Concept", "三元组", "知识的基本表示单位:<主体, 谓词, 客体>"),
    ("e-neo4j", "Technology", "Neo4j", "原生图数据库,使用Cypher查询语言,支持BFS/DFS图算法"),
    ("e-cypher", "Technology", "Cypher", "Neo4j的声明式图查询语言,用于模式匹配和图遍历"),
    ("e-pgvector", "Technology", "PGVector", "PostgreSQL的向量搜索扩展,支持余弦相似度等距离度量"),
    ("e-rag", "Concept", "RAG", "检索增强生成,结合信息检索和文本生成减少LLM幻觉"),
    ("e-embedding", "Concept", "Embedding", "将非结构化数据转换为高维向量的技术"),
    ("e-bfs", "Algorithm", "BFS", "广度优先搜索,用于知识图谱的多跳遍历"),
    ("e-hungarian", "Algorithm", "匈牙利算法", "解决二部图最优匹配问题的经典算法"),
    ("e-fast-rcnn", "Model", "Fast R-CNN", "两阶段目标检测模型:先生成Proposal再分类"),
    ("e-detr", "Model", "DETR", "端到端目标检测模型,使用匈牙利算法进行预测匹配"),
    ("e-langgraph", "Technology", "LangGraph", "基于有向图的Agent编排框架,支持状态管理和HITL"),
    ("e-multi-agent", "Concept", "多Agent系统", "由多个自主Agent协同工作的系统架构"),
    ("e-google", "Organization", "Google", "2012年首次提出知识图谱概念用于增强搜索"),
    ("e-proposal", "Concept", "Proposal", "将生成任务分解为多个同结构子任务的机制"),
    ("e-cascade", "Algorithm", "Cascade融合", "多轮迭代式检索结果融合策略"),
    ("e-hitl", "Concept", "HITL", "Human-In-The-Loop,人工介入AI决策流程的模式"),
    ("e-scoring", "Concept", "去噪得分函数", "结合自监督去噪和稀疏矩阵的非对称评分方法"),
]

KG_RELATIONS = [
    ("e-kg", "COMPOSED_OF", "e-entity", "知识图谱由实体构成"),
    ("e-kg", "COMPOSED_OF", "e-relation", "知识图谱由关系构成"),
    ("e-entity", "REPRESENTED_AS", "e-triple", "实体通过三元组表示"),
    ("e-relation", "REPRESENTED_AS", "e-triple", "关系通过三元组表示"),
    ("e-kg", "STORED_IN", "e-neo4j", "知识图谱可存储在Neo4j中"),
    ("e-neo4j", "USES", "e-cypher", "Neo4j使用Cypher查询语言"),
    ("e-neo4j", "SUPPORTS", "e-bfs", "Neo4j原生支持BFS遍历"),
    ("e-rag", "USES", "e-embedding", "RAG使用Embedding进行语义检索"),
    ("e-rag", "USES", "e-pgvector", "RAG使用PGVector存储向量"),
    ("e-rag", "ENHANCED_BY", "e-kg", "RAG可通过知识图谱增强深度推理"),
    ("e-fast-rcnn", "INSPIRES", "e-proposal", "Fast R-CNN的Proposal思想迁移到生成任务"),
    ("e-detr", "USES", "e-hungarian", "DETR使用匈牙利算法进行匹配"),
    ("e-hungarian", "APPLIED_IN", "e-scoring", "匈牙利算法应用于评估匹配"),
    ("e-langgraph", "ORCHESTRATES", "e-multi-agent", "LangGraph编排多Agent系统"),
    ("e-multi-agent", "INCLUDES", "e-rag", "多Agent系统包含检索Agent(RAG)"),
    ("e-multi-agent", "INCLUDES", "e-proposal", "多Agent系统包含生成Agent(Proposal)"),
    ("e-multi-agent", "INCLUDES", "e-hitl", "多Agent系统包含HITL机制"),
    ("e-google", "CREATED", "e-kg", "Google于2012年提出知识图谱概念"),
    ("e-cascade", "FUSES", "e-rag", "Cascade融合RAG检索结果"),
    ("e-scoring", "EVALUATES", "e-proposal", "去噪得分函数评估Proposal生成质量"),
]


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call GLM embedding-3 API to get embeddings."""
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for embedding generation")

    url = API_BASE.rstrip("/") + "/embeddings"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBEDDING_MODEL, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


async def seed_pgvector():
    """Load documents into PGVector."""
    print("\n=== Seeding PGVector ===")

    # Connect
    conn = await psycopg.AsyncConnection.connect(PG_DSN, autocommit=True)

    # Ensure table
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {COLLECTION} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    print(f"  Table '{COLLECTION}' ensured.")

    # Get embeddings in batches
    contents = [d["content"] for d in DOCUMENTS]
    print(f"  Generating embeddings for {len(contents)} documents...")

    batch_size = 5
    all_embeddings = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        embeddings = await get_embeddings(batch)
        all_embeddings.extend(embeddings)
        print(f"    Batch {i // batch_size + 1}: {len(batch)} docs embedded")

    # Upsert
    for doc, emb in zip(DOCUMENTS, all_embeddings):
        emb_str = str(emb)
        meta_str = json.dumps(doc["metadata"])
        await conn.execute(
            f"""
            INSERT INTO {COLLECTION} (id, content, embedding, metadata)
            VALUES (%s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
            """,
            (doc["id"], doc["content"], emb_str, meta_str),
        )

    print(f"  Inserted/updated {len(DOCUMENTS)} documents.")

    # Verify
    cur = await conn.execute(f"SELECT count(*) FROM {COLLECTION}")
    row = await cur.fetchone()
    print(f"  Total documents in table: {row[0]}")

    await conn.close()


async def seed_neo4j():
    """Load entities and relations into Neo4j."""
    print("\n=== Seeding Neo4j ===")

    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    async with driver.session(database="neo4j") as session:
        # Clear existing data (dev only)
        await session.run("MATCH (n) DETACH DELETE n")
        print("  Cleared existing data.")

        # Create entities
        for eid, label, name, description in KG_ENTITIES:
            await session.run(
                f"CREATE (n:{label} {{id: $id, name: $name, description: $desc}})",
                {"id": eid, "name": name, "desc": description},
            )
        print(f"  Created {len(KG_ENTITIES)} entities.")

        # Create relations
        for src, rel_type, tgt, desc in KG_RELATIONS:
            await session.run(
                f"""
                MATCH (a {{id: $src}}), (b {{id: $tgt}})
                CREATE (a)-[:{rel_type} {{description: $desc}}]->(b)
                """,
                {"src": src, "tgt": tgt, "desc": desc},
            )
        print(f"  Created {len(KG_RELATIONS)} relations.")

        # Verify
        result = await session.run("MATCH (n) RETURN count(n) AS cnt")
        data = await result.data()
        print(f"  Total entities: {data[0]['cnt']}")

        result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        data = await result.data()
        print(f"  Total relations: {data[0]['cnt']}")

    await driver.close()


async def main():
    print("=" * 60)
    print("haozhong Demo Data Seeder")
    print("=" * 60)

    await seed_pgvector()
    await seed_neo4j()

    print("\n" + "=" * 60)
    print("Seeding complete! Try:")
    print('  curl -X POST http://localhost:8000/api/v1/agent/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "什么是知识图谱,它有哪些应用?"}\'')
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
