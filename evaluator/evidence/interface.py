"""Types every M0 implementation is expressed in.

Designed against the three implementations as they exist, not as a clean-slate
abstraction. Two properties of the real code drove every decision here:

**Extractors disagree about what an evidence item is.** The recognizer emits
curve-grid intersections in benchmark units; the presenter emits measured points
at LLM-supplied pixels; the PlotQA chain emits plotted marks carrying a series
and a category label. Forcing one schema on all three would either discard the
category labels or invent coordinates the PlotQA chain never computes. So
:class:`EvidenceItem` carries the fields all three genuinely produce and leaves
the rest in ``extra``, and :attr:`EvidenceBundle.item_semantics` names what the
items mean rather than pretending the three are interchangeable in meaning.

**Every extractor has preconditions, and they are not the same ones.** The grid
recognizer needs at least three grid lines per axis; the ruler-based paths need
readable tick labels. That is why :meth:`EvidenceProvider.applicability` exists
as a separate method: it makes "this chart is outside my domain" a first-class,
inspectable answer rather than an empty result indistinguishable from failure.

Nothing here scores, ranks, or filters evidence. Reliability scoring and
evidence selection were retired on 2026-09-19 and must not re-enter through this
interface under another name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class Applicability(str, Enum):
    """Whether a provider's structural preconditions hold for one chart.

    ``UNKNOWN`` is distinct from ``UNSUPPORTED`` on purpose. A provider that
    cannot decide without running the full extraction says ``UNKNOWN``; one that
    has positively determined the chart lacks what it needs says
    ``UNSUPPORTED``. Collapsing the two would turn "I did not check" into "it
    does not apply", which is how an untested claim becomes a reported fact.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ApplicabilityReport:
    """A provider's verdict on one chart, with the reason it reached it."""

    verdict: Applicability
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.verdict is Applicability.SUPPORTED


@dataclass(frozen=True)
class ChartEvidenceRequest:
    """One chart plus whatever auxiliary input a provider may need.

    The auxiliary fields are optional because no provider needs all of them, and
    a provider that needs one it did not get must report that through
    :meth:`EvidenceProvider.applicability` rather than fabricating a substitute.

    ``ocr_row`` holds pre-computed OCR boxes. It is an input rather than
    something providers fetch, because the PlotQA runs read frozen boxes from
    disk and re-running OCR there would both cost money and break reproduction
    of the frozen artifacts.
    """

    image_path: Path
    sample_id: str = ""
    question: str = ""
    ocr_row: Optional[Dict[str, Any]] = None
    located_points: Optional[List[Dict[str, Any]]] = None
    record: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    """One extracted fact, in the fields all three implementations produce.

    ``value_x`` / ``value_y`` are chart-space values and are ``None`` when the
    provider could not map pixels to values. ``None`` is the interface's way of
    saying "unmeasured": no provider may substitute a guess for it, because a
    guessed value is indistinguishable downstream from a measured one.
    """

    value_x: Optional[float] = None
    value_y: Optional[float] = None
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None
    series_label: Optional[str] = None
    category_label: Optional[str] = None
    item_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """True when the item carries at least one chart-space value."""
        return self.value_x is not None or self.value_y is not None


@dataclass(frozen=True)
class EvidenceBundle:
    """What one provider extracted from one chart.

    ``succeeded`` is not ``bool(items)``. A provider that ran correctly on a
    chart with nothing plottable succeeds with zero items, and one that failed
    mid-pipeline fails even if partial items survive. Conflating the two hides
    extraction failures as empty charts.
    """

    provider: str
    applicability: ApplicabilityReport
    items: List[EvidenceItem] = field(default_factory=list)
    item_semantics: str = "unspecified"
    axis_model: Dict[str, Any] = field(default_factory=dict)
    succeeded: bool = False
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def measured_items(self) -> List[EvidenceItem]:
        return [item for item in self.items if item.measured]


@runtime_checkable
class EvidenceProvider(Protocol):
    """The contract an M0 implementation satisfies.

    Deliberately three members. Anything more would describe one implementation
    rather than the family: the grid recognizer has no tick model to expose, the
    PlotQA chain has no notion of a located point to measure at.
    """

    name: str
    #: Chart structures this provider is built for, for reporting and for
    #: reviewers checking that a provider was not run outside its domain.
    domain: str

    def applicability(self, request: ChartEvidenceRequest) -> ApplicabilityReport:
        """Decide whether this chart meets the provider's preconditions."""

    def extract(self, request: ChartEvidenceRequest) -> EvidenceBundle:
        """Extract evidence, or return a failed bundle explaining why not."""


def unsupported_bundle(
    provider: str,
    report: ApplicabilityReport,
    *,
    item_semantics: str = "unspecified",
) -> EvidenceBundle:
    """Build the empty bundle a provider returns outside its domain.

    Shared so that "out of domain" is one recognisable shape rather than three
    near-identical ones that drift apart.
    """
    return EvidenceBundle(
        provider=provider,
        applicability=report,
        items=[],
        item_semantics=item_semantics,
        succeeded=False,
        failure_reason=f"not_applicable:{report.reason}",
    )
