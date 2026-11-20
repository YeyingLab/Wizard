"""Stage 4 entry point of the released chain: `wizard.distill.dmd` unchanged, with PAST + FUTURE memory candidates as the causal
teacher and the consistency student were trained (`wizard.distill.consistency_selfaug_future`): SceneFly forward windows keep memory
picks >= 3 s before the window or >= 3 s after its end. Use with --mem_index mem_index_both.pt (roaming picks from both directions).
Inference is unchanged (past memory only).
   arguments and environment exactly as wizard.distill.dmd"""
import os, json
import wizard.train.causal as A
class ARDataFuture(A.ClipDataset):
    def __init__(self, root, rows, a):
        A.ClipDataset.__init__(self, root, rows, a)
        info = json.load(open(f"{A.SFLY}/index.json")) if os.path.exists(f"{A.SFLY}/index.json") else {}
        sf = A.lt(f"{A.SFLY}/mem_index.pt") if os.path.exists(f"{A.SFLY}/mem_index.pt") else {}
        n0 = n1 = 0
        for name in sf:
            if name not in info or name not in self.mem_index: continue
            f0 = int(info[name]["f0"]); e = dict(self.mem_index[name])
            ch = [[(f, g) for f, g in c if f <= f0 - A.GAP_F or f >= f0 + 121 + A.GAP_F] for c in e["chunks"]]
            n0 += sum(len(c) for c in e["chunks"]); n1 += sum(len(c) for c in ch); e["chunks"] = ch; self.mem_index[name] = e
        self.past_info = (n0, n1); print(f"MEM_FUTURE SceneFly picks kept (>= 3 s before or after the window): {n1} of {n0}", flush=True)
A.ARData = ARDataFuture
import wizard.distill.dmd as D
if __name__ == "__main__": D.main()
