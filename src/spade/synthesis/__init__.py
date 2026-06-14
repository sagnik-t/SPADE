"""Stage III synthesis: inference-only assembly of the frozen stage models.

Separate from :mod:`spade.models` because nothing here is trained — it composes
the frozen Stage I gate/decoder and Stage II generators into a pipeline that
emits a discrete :class:`SyntheticDataset`. :class:`SynthesisModel` runs the
expand → retrieve → gate → decode flow; :func:`top_c_candidates` is the faiss
sparse-candidate retrieval that keeps it sub-quadratic.
"""

from spade.synthesis.ann import top_c_candidates
from spade.synthesis.dataset import SyntheticDataset
from spade.synthesis.synthesizer import SynthesisModel

__all__ = ["SynthesisModel", "SyntheticDataset", "top_c_candidates"]
