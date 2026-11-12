from dataclasses import dataclass

@dataclass(frozen=True)
class Decision:
    route: str
    threshold: float
    review: bool

def decide(score: float, threshold: float=.63) -> Decision:
    review=float(score)>=threshold
    return Decision("HUMAN_REVIEW" if review else "AOS",threshold,review)

def optimize_roi(labels,scores,avg_fraud_loss,review_cost,aos_saving):
    best=None
    for n in range(1,100):
        t=n/100; benefit=0.
        for y,s in zip(labels,scores):
            review=s>=t
            benefit += (avg_fraud_loss-review_cost if y and review else -avg_fraud_loss if y else -review_cost if review else aos_saving)
        if best is None or benefit>best[1]: best=(t,benefit)
    return {"threshold":best[0],"expected_net_benefit":best[1]}
