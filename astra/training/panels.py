"""Strict ordered input/output gene identities."""
from dataclasses import dataclass

@dataclass(frozen=True)
class GenePanels:
    input_gene_ids: tuple[str, ...]
    output_gene_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("input_gene_ids", "output_gene_ids"):
            genes = tuple(getattr(self, name))
            if not genes or any(not isinstance(g, str) or not g.strip() for g in genes):
                raise ValueError(f"{name} must contain nonempty gene IDs")
            if len(set(genes)) != len(genes):
                raise ValueError(f"{name} contains duplicate gene IDs")
            object.__setattr__(self, name, genes)
        if not set(self.output_gene_ids).issubset(self.input_gene_ids):
            raise ValueError("output genes must be a subset of input genes for exact conservation")

    @property
    def output_indices(self):
        positions = {gene: i for i, gene in enumerate(self.input_gene_ids)}
        return tuple(positions[gene] for gene in self.output_gene_ids)

    def as_dict(self):
        return dict(input_gene_ids=list(self.input_gene_ids), output_gene_ids=list(self.output_gene_ids))
