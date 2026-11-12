from dataclasses import dataclass

@dataclass(frozen=True)
class FusionResult:
    score: float
    dominant_model: str

def fuse(scores: dict[str,float], weights=None) -> FusionResult:
    weights=weights or {"trufor":.35,"fused":.50,"recapture":.15}
    clean={k:max(0.,min(1.,float(scores[k]))) for k in weights}
    value=sum(clean[k]*weights[k] for k in weights)/sum(weights.values())
    return FusionResult(round(value,6),max(clean,key=clean.get))
