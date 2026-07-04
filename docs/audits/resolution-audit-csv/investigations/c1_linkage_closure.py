"""C1 proof artifact: single-linkage cannot separate cancel-subscription vs cancel-order.

Reproducible closure of the INVESTIGATIONS.md C1 claim. The negative cosine separation gap
alone proves only PAIRWISE inseparability; this computes the single-linkage MST connectivity
bottleneck vs the inter-intent merge distance to settle single-linkage separability.

Run: python docs/audits/resolution-audit-csv/investigations/c1_linkage_closure.py
Deps: sentence-transformers, scipy (audit venv). Deterministic (fixed phrasings + model).
"""
import itertools
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

SUB = ["cancel my subscription", "how do I cancel my monthly plan", "stop my recurring billing",
       "end my membership", "turn off auto-renew", "i want to cancel the subscription please",
       "unsubscribe me from the paid plan", "cancel recurring payments", "close my subscription",
       "stop charging me every month"]
ORDER = ["cancel my order", "cancel order #12345", "i need to cancel a purchase",
         "stop my order from shipping", "cancel the item I just bought", "how do I cancel an order",
         "please cancel my recent order", "void my order", "cancel this purchase", "kill order 998"]


def main() -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    labels = np.array([0] * 10 + [1] * 10)
    emb = model.encode(SUB + ORDER, normalize_embeddings=True)
    sim = emb @ emb.T

    inter = [sim[i, j] for i in range(10) for j in range(10, 20)]
    intra = ([sim[i, j] for i, j in itertools.combinations(range(10), 2)]
             + [sim[i, j] for i, j in itertools.combinations(range(10, 20), 2)])
    print(f"intra sim: min={min(intra):.3f} max={max(intra):.3f}")
    print(f"inter sim: min={min(inter):.3f} max={max(inter):.3f}")
    print(f"separation gap = {min(intra) - max(inter):.3f}  (negative => pairwise bands overlap)")

    merge_dist = 1.0 - max(inter)                       # single-linkage joins the two intents here

    def mst_bottleneck(idx: range) -> float:            # distance where the intent becomes one component
        z = linkage(pdist(emb[idx], "cosine"), method="single")
        return float(z[:, 2].max())

    connect_dist = max(mst_bottleneck(range(10)), mst_bottleneck(range(10, 20)))
    print(f"\nmerge_distance   (intents join)       = 1 - max_inter = {merge_dist:.4f}")
    print(f"connect_distance (both intents whole) = max MST bottleneck = {connect_dist:.4f}")

    z_all = linkage(pdist(emb, "cosine"), method="single")
    pure_band = [t for t in np.linspace(0.01, 0.99, 197)
                 if len(set(fcluster(z_all, t, "distance"))) == 2
                 and np.array_equal(fcluster(z_all, t, "distance") == fcluster(z_all, t, "distance")[0],
                                    labels == labels[0])]
    if pure_band:
        print(f"\nPURE-2 band exists for T in [{min(pure_band):.3f}, {max(pure_band):.3f}] "
              "=> single-linkage is threshold-FRAGILE, not impossible.")
    else:
        print("\nNo pure-2 band at any single-linkage threshold => IMPOSSIBILITY holds for this fixture.")
    print(f"connect >= merge ? {connect_dist:.4f} >= {merge_dist:.4f} = {connect_dist >= merge_dist}")


if __name__ == "__main__":
    main()
