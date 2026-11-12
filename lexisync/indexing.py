import re, time
from collections import defaultdict
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+(?:\.[0-9]+)*", text.lower())

class HybridIndex:
    def __init__(self, chunks):
        self.chunks=chunks
        if not chunks: raise ValueError("No text chunks to index")
        self.tokens=[tokenize(c["text"]+" "+str(c.get("section",""))) for c in chunks]
        self.bm25=BM25Okapi(self.tokens)
        self.embedder=None; self.cross=None; self.vectors=None
        self.embedding_status="not loaded"; self.reranker_status="not loaded"
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.vectors=self.embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
            self.embedding_status="all-MiniLM-L6-v2 loaded"
        except Exception as e: self.embedding_status=f"unavailable: {e}"
        try:
            from sentence_transformers import CrossEncoder
            self.cross=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker_status="ms-marco-MiniLM-L-6-v2 loaded"
        except Exception as e: self.reranker_status=f"unavailable: {e}"

    def search(self, query, top_k=6, mode="hybrid", rerank=True):
        t=time.perf_counter()
        qtok=tokenize(query)
        bm=self.bm25.get_scores(qtok)
        bm_order=sorted(range(len(bm)),key=lambda i:bm[i],reverse=True)
        dense_order=[]
        dense_scores=None
        if self.embedder is not None and self.vectors is not None:
            import numpy as np
            qv=self.embedder.encode([query],normalize_embeddings=True)[0]
            dense_scores=np.dot(self.vectors,qv)
            dense_order=sorted(range(len(dense_scores)),key=lambda i:float(dense_scores[i]),reverse=True)
        if mode=="dense":
            if dense_scores is None:
                raise RuntimeError("Dense retrieval is unavailable because the embedding model did not load.")
            base_order=dense_order
            base_scores=dense_scores
        elif mode=="bm25":
            base_order=bm_order
            base_scores=bm
        else:
            # Reciprocal rank fusion, retaining exact lexical matches.
            scores=defaultdict(float)
            for rank,i in enumerate(bm_order[:50]): scores[i]+=1/(60+rank)
            for rank,i in enumerate(dense_order[:50]): scores[i]+=1/(60+rank)
            base_order=sorted(scores,key=scores.get,reverse=True)
            base_scores=scores
        candidate_limit=max(top_k*10,50)
        candidates=base_order[:candidate_limit]
        reranker_used=bool(rerank and self.cross and candidates)
        if reranker_used:
            pairs=[(query,self.chunks[i]["text"]) for i in candidates]
            cross_scores=self.cross.predict(pairs)
            cross_score_by_id={i:float(score) for i,score in zip(candidates,cross_scores)}
            cross_order=sorted(candidates,key=cross_score_by_id.get,reverse=True)
            base_rank={i:rank for rank,i in enumerate(candidates)}
            cross_rank={i:rank for rank,i in enumerate(cross_order)}
            fused_order=sorted(
                candidates,
                key=lambda i:1/(60+base_rank[i])+1/(60+cross_rank[i]),
                reverse=True,
            )
            ranked=[(i,cross_score_by_id[i]) for i in fused_order[:top_k]]
        else:
            ranked=[(i,float(base_scores[i])) for i in candidates[:top_k]]
        results=[]
        for i,score in ranked:
            c=dict(self.chunks[i]); c["score"]=score; results.append(c)
        return results, (time.perf_counter()-t)*1000, {"mode":mode,"bm25_candidates":len(bm_order[:50]),
            "dense_enabled":dense_scores is not None,"reranker_enabled":bool(self.cross),
            "candidate_pool_size":len(candidates),"reranker_fused":reranker_used}
