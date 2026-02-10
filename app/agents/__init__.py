"""Business worker layer — multi-agent system.

Three independent agent stages with belief+intent evidence graph interaction:
- RetrievalAgent: query decomposition (breadth) + KG BFS (depth) + cascade fusion
- GenerationAgent: proposal-based template extraction (Fast RCNN inspired)
- EvaluationAgent: Hungarian matching scoring + auto prompt optimization
"""
