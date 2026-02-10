"""Generation Agent -- evidence graph slot-filling via proposals.

Two-stage pipeline inspired by Fast RCNN:
- Stage 1 (Proposal Decomposer): generate N same-structured slot-filling proposals
- Stage 2 (Content Generator): EXTRACT from evidence graph to fill each proposal's slots

Key difference from template mode:
- Proposals define SLOTS (structured extraction targets), not templates
- Each slot is filled by extracting from evidence graph nodes
- All proposals share the same structure = consistent granularity
"""

from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.generation.content_generator import ContentGenerator
from app.agents.generation.proposal_decomposer import ProposalDecomposer
from app.core.config import settings
from app.core.logging import logger
from app.core.types.agent_io import AgentInput, AgentOutput, AgentStatus


class GenerationAgent(BaseAgent):
    """Two-stage generation: propose N same-type items -> slot-fill from evidence graph."""

    agent_name = "generation_agent"

    def __init__(self) -> None:
        self.proposal_decomposer = ProposalDecomposer(
            max_proposals=settings.GENERATION_MAX_PROPOSALS,
        )
        self.content_generator = ContentGenerator()

    async def _execute(self, agent_input: AgentInput) -> AgentOutput:
        """Run the two-stage evidence graph slot-filling pipeline.

        1) Proposal Decomposition: generate N same-structured proposals,
           each defining slots to extract from evidence graph
        2) Slot Extraction: for each proposal, extract evidence to fill slots
        3) Assemble final output from slot-filled sections
        """
        query = agent_input.query
        evidence_graph = agent_input.evidence_graph

        logger.info("generation_pipeline_start", query=query)

        # -- Stage 1: Proposal Decomposition (RPN analogue) ---------------
        # Generate N same-structured slot-filling proposals
        proposal_nodes = await self.proposal_decomposer.generate_proposals(
            query, evidence_graph
        )

        if not proposal_nodes:
            return AgentOutput(
                agent_name=self.agent_name,
                status=AgentStatus.ERROR,
                error_message="No proposals generated",
                summary="Failed to decompose generation task into proposals.",
            )

        # -- Stage 2: Slot Extraction (ROI Head analogue) -----------------
        # Fill each proposal's slots by extracting from evidence graph
        generated_nodes = await self.content_generator.generate(
            proposal_nodes, evidence_graph, query
        )

        # -- Assemble final output ----------------------------------------
        final_text = self._assemble_output(generated_nodes)

        all_new_nodes = proposal_nodes + generated_nodes

        status = (
            AgentStatus.SUCCESS if generated_nodes
            else AgentStatus.PARTIAL
        )

        logger.info(
            "generation_pipeline_complete",
            proposals=len(proposal_nodes),
            generated=len(generated_nodes),
            status=status.value,
        )

        return AgentOutput(
            agent_name=self.agent_name,
            status=status,
            new_nodes=all_new_nodes,
            summary=(
                f"Slot-filled {len(generated_nodes)} sections "
                f"from {len(proposal_nodes)} proposals."
            ),
            metrics={
                "proposal_count": len(proposal_nodes),
                "generated_count": len(generated_nodes),
                "final_output_length": len(final_text),
            },
        )

    @staticmethod
    def _assemble_output(generated_nodes) -> str:
        """Assemble slot-filled sections into final output text."""
        if not generated_nodes:
            return ""

        sections = []
        for node in generated_nodes:
            # New metadata uses "aspect" instead of old "title"
            aspect = node.evidence.metadata.get("aspect", "")
            content = node.evidence.content
            if aspect:
                sections.append(content)
            else:
                sections.append(content)

        return "\n\n---\n\n".join(sections)
