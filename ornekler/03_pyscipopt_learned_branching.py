"""PySCIPOpt + strong branching + bipartite GNN learned branching örneği.

Bu dosya gerçek SCIP branching API'sini kullanır:
- getLPBranchCands()
- startStrongbranch() / getVarStrongbranch() / endStrongbranch()
- custom Branchrule

Akış:
1) küçük Maximum Independent Set (MIS) MILP'leri oluştur,
2) strong branching'i expert olarak kullanıp solver-state etiketleri topla,
3) her state'i variable-constraint bipartite graph'a çevir,
4) bipartite GNN'yi expert kararını taklit edecek şekilde eğit,
5) eğitilmiş GNN'yi yeni bir SCIP branching rule içinde kullan.

Bu, Gasse vd. (NeurIPS 2019) çalışmasının birebir reprodüksiyonu değildir;
ancak aynı temel fikri eğitim amaçlı küçük bir örnekte uygular:
https://arxiv.org/abs/1906.01629

PySCIPOpt branching dokümantasyonu:
https://pyscipopt.readthedocs.io/en/latest/tutorials/branchrule.html
"""

import random
import numpy as np
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F

from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_PARAMSETTING
from torch_geometric.data import HeteroData

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_mis_instance(n=14, p=0.28, seed=0):
    """Rastgele ağırlıklı MIS örneği üretir.

    LP relaxation'ın branching üretmesini kolaylaştırmak için en az bir odd cycle
    graph'a eklenir.
    """
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(n=n, p=p, seed=seed)

    if n >= 5:
        cycle = [0, 1, 2, 3, 4]
        for u, v in zip(cycle, cycle[1:] + cycle[:1]):
            graph.add_edge(u, v)

    weights = rng.integers(1, 11, size=n).astype(float)
    return graph, weights


def build_scip_mis(graph, weights, name="mis"):
    """MIS'i eşdeğer minimizasyon MILP'i olarak kurar.

    max sum(w_i x_i) yerine min -sum(w_i x_i) kullanılır. Böylece strong
    branching dual-bound gain hesabı minimizasyon yönünde okunabilir.
    """
    scip = Model(name)

    # Eğitim örneğinde graph/state eşlemesini okunabilir tutmak için kapatılıyor.
    # Production benchmark'ta bu ayarlar ayrıca karşılaştırılmalıdır.
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    scip.hideOutput()

    variables = []
    for i, weight in enumerate(weights):
        var = scip.addVar(name=f"x_{i}", vtype="B", obj=-float(weight))
        variables.append(var)

    edges = list(graph.edges())
    for k, (u, v) in enumerate(edges):
        scip.addCons(
            variables[u] + variables[v] <= 1,
            name=f"edge_{k}_{u}_{v}",
        )

    scip.setMinimize()
    return scip, variables, edges


def fractionality(value):
    """0 -> integral, 1 -> tam 0.5 fractional olacak şekilde ölçekler."""
    return float(min(value - np.floor(value), np.ceil(value) - value) * 2.0)


def build_bipartite_state(
    scip,
    graph,
    weights,
    edges,
    candidate_names=None,
):
    """Mevcut SCIP LP durumunu PyG HeteroData'ya çevirir.

    variable features:
      objective coefficient, LP value, fractionality,
      local LB, local UB, graph degree

    constraint features:
      RHS, current LP slack, degree

    edge feature:
      linear coefficient (MIS için 1)
    """
    transformed = {v.name: v for v in scip.getVars(transformed=True)}
    n_vars = len(weights)

    lp_vals = np.zeros(n_vars, dtype=float)
    lbs = np.zeros(n_vars, dtype=float)
    ubs = np.ones(n_vars, dtype=float)

    for i in range(n_vars):
        var = transformed[f"x_{i}"]
        lp_vals[i] = float(var.getLPSol())
        lbs[i] = float(var.getLbLocal())
        ubs[i] = float(var.getUbLocal())

    obj = -np.asarray(weights, dtype=float)
    obj_scale = max(np.max(np.abs(obj)), 1.0)
    degrees = np.asarray([graph.degree(i) for i in range(n_vars)], dtype=float)
    degree_scale = max(np.max(degrees), 1.0)

    var_x = np.column_stack(
        [
            obj / obj_scale,
            lp_vals,
            [fractionality(x) for x in lp_vals],
            lbs,
            ubs,
            degrees / degree_scale,
        ]
    ).astype(np.float32)

    con_x = []
    edge_src = []
    edge_dst = []
    edge_attr = []

    for j, (u, v) in enumerate(edges):
        slack = 1.0 - lp_vals[u] - lp_vals[v]
        con_x.append([1.0, slack, 2.0 / max(n_vars, 1)])

        for node in (u, v):
            edge_src.append(node)
            edge_dst.append(j)
            edge_attr.append([1.0])

    data = HeteroData()
    data["variable"].x = torch.tensor(var_x, dtype=torch.float32)
    data["constraint"].x = torch.tensor(np.asarray(con_x), dtype=torch.float32)

    edge_store = data["variable", "participates", "constraint"]
    edge_store.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_store.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    candidate_mask = torch.zeros(n_vars, dtype=torch.bool)
    if candidate_names is not None:
        for name in candidate_names:
            idx = int(name.split("_")[1])
            candidate_mask[idx] = True

    data["variable"].candidate_mask = candidate_mask
    return data


class StrongBranchCollector(Branchrule):
    """Strong branching'i expert policy olarak kullanıp imitation labels toplar."""

    def __init__(self, graph, weights, edges, itlim=100):
        self.graph = graph
        self.weights = np.asarray(weights, dtype=float)
        self.edges = edges
        self.itlim = itlim
        self.samples = []

    def branchexeclp(self, allowaddcons):
        (
            branch_cands,
            branch_cand_sols,
            branch_cand_fracs,
            ncands,
            npriocands,
            nimplcands,
        ) = self.model.getLPBranchCands()

        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        lpobj = float(self.model.getLPObjVal())
        scores = np.full(npriocands, -np.inf, dtype=float)
        lperror = False

        self.model.startStrongbranch()
        try:
            for i in range(npriocands):
                (
                    down,
                    up,
                    downvalid,
                    upvalid,
                    downinf,
                    upinf,
                    downconflict,
                    upconflict,
                    this_lperror,
                ) = self.model.getVarStrongbranch(
                    branch_cands[i],
                    self.itlim,
                    idempotent=True,
                )

                if this_lperror:
                    lperror = True
                    break

                down_gain = (
                    max(float(down) - lpobj, 0.0) if downvalid else 0.0
                )
                up_gain = max(float(up) - lpobj, 0.0) if upvalid else 0.0

                # Infeasible child güçlü sinyaldir; eğitim örneğinde büyük gain veriyoruz.
                if downinf:
                    down_gain = max(down_gain, 1e6)
                if upinf:
                    up_gain = max(up_gain, 1e6)

                scores[i] = float(
                    self.model.getBranchScoreMultiple(
                        branch_cands[i], [down_gain, up_gain]
                    )
                )
        finally:
            self.model.endStrongbranch()

        if lperror or not np.isfinite(scores).any():
            return {"result": SCIP_RESULT.DIDNOTRUN}

        candidate_names = [branch_cands[i].name for i in range(npriocands)]
        best_local = int(np.nanargmax(scores))
        best_name = candidate_names[best_local]
        best_global = int(best_name.split("_")[1])

        data = build_bipartite_state(
            self.model,
            self.graph,
            self.weights,
            self.edges,
            candidate_names=candidate_names,
        )
        data["variable"].target_index = torch.tensor(
            [best_global], dtype=torch.long
        )

        expert_scores = torch.full(
            (len(self.weights),), float("-inf"), dtype=torch.float32
        )
        for name, score in zip(candidate_names, scores):
            idx = int(name.split("_")[1])
            expert_scores[idx] = float(score)
        data["variable"].expert_scores = expert_scores

        self.samples.append(data.cpu())

        # Expert kararını uygula; böylece B&B ağacında veri toplamaya devam edilir.
        self.model.branchVarVal(
            branch_cands[best_local],
            branch_cand_sols[best_local],
        )
        return {"result": SCIP_RESULT.BRANCHED}


def collect_from_instance(seed, n=14, p=0.28):
    graph, weights = make_mis_instance(n=n, p=p, seed=seed)
    scip, variables, edges = build_scip_mis(graph, weights, name=f"mis_{seed}")

    collector = StrongBranchCollector(graph, weights, edges, itlim=80)
    scip.includeBranchrule(
        collector,
        name="strong_branch_collector",
        desc="Strong branching expert labels for GNN imitation learning",
        priority=1_000_000,
        maxdepth=-1,
        maxbounddist=1.0,
    )

    scip.optimize()

    stats = {
        "status": str(scip.getStatus()),
        "nodes": int(scip.getNTotalNodes()),
        "time": float(scip.getSolvingTime()),
        "samples": len(collector.samples),
    }
    return collector.samples, stats


def mean_aggregate(messages, index, dim_size):
    out = messages.new_zeros((dim_size, messages.size(-1)))
    out.index_add_(0, index, messages)

    count = messages.new_zeros((dim_size, 1))
    count.index_add_(0, index, messages.new_ones((messages.size(0), 1)))
    return out / count.clamp_min(1.0)


class BipartiteBranchNet(nn.Module):
    """variables -> constraints -> variables message passing."""

    def __init__(self, var_dim=6, con_dim=3, hidden=64):
        super().__init__()
        self.var_enc = nn.Sequential(
            nn.Linear(var_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.con_enc = nn.Sequential(
            nn.Linear(con_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        self.v_to_c = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.c_upd = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        self.c_to_v = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.v_upd = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        self.score = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):
        h_var = self.var_enc(data["variable"].x)
        h_con = self.con_enc(data["constraint"].x)

        store = data["variable", "participates", "constraint"]
        var_idx, con_idx = store.edge_index
        edge_attr = store.edge_attr

        msg_vc = self.v_to_c(torch.cat([h_var[var_idx], edge_attr], dim=-1))
        agg_c = mean_aggregate(msg_vc, con_idx, h_con.size(0))
        h_con = h_con + self.c_upd(torch.cat([h_con, agg_c], dim=-1))

        msg_cv = self.c_to_v(torch.cat([h_con[con_idx], edge_attr], dim=-1))
        agg_v = mean_aggregate(msg_cv, var_idx, h_var.size(0))
        h_var = h_var + self.v_upd(torch.cat([h_var, agg_v], dim=-1))

        return self.score(h_var).squeeze(-1)


def masked_imitation_loss(model, data):
    data = data.to(device)
    logits = model(data)

    mask = data["variable"].candidate_mask
    target = int(data["variable"].target_index.item())

    masked_logits = logits.masked_fill(~mask, -1e9).unsqueeze(0)
    target_tensor = torch.tensor([target], device=device)
    return F.cross_entropy(masked_logits, target_tensor)


def collect_dataset(num_instances=10):
    samples = []
    stats = []

    for seed in range(num_instances):
        instance_samples, instance_stats = collect_from_instance(
            seed=100 + seed,
            n=14,
            p=0.28,
        )
        samples.extend(instance_samples)
        stats.append(instance_stats)

    return samples, stats


def train_model(train_samples, epochs=40):
    model = BipartiteBranchNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5
    )

    for epoch in range(epochs):
        random.shuffle(train_samples)
        total = 0.0

        for sample in train_samples:
            optimizer.zero_grad()
            loss = masked_imitation_loss(model, sample)
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            denom = max(len(train_samples), 1)
            print(f"epoch={epoch:02d} loss={total / denom:.4f}")

    return model


class GNNBranchRule(Branchrule):
    """Strong branching çağırmadan eğitilmiş GNN ile branching yapar."""

    def __init__(self, graph, weights, edges, torch_model):
        self.graph = graph
        self.weights = np.asarray(weights, dtype=float)
        self.edges = edges
        self.torch_model = torch_model

    def branchexeclp(self, allowaddcons):
        (
            branch_cands,
            branch_cand_sols,
            branch_cand_fracs,
            ncands,
            npriocands,
            nimplcands,
        ) = self.model.getLPBranchCands()

        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        candidate_names = [branch_cands[i].name for i in range(npriocands)]
        data = build_bipartite_state(
            self.model,
            self.graph,
            self.weights,
            self.edges,
            candidate_names=candidate_names,
        ).to(device)

        self.torch_model.eval()
        with torch.no_grad():
            logits = self.torch_model(data)
            logits = logits.masked_fill(
                ~data["variable"].candidate_mask,
                -1e9,
            )
            best_global = int(torch.argmax(logits).item())

        best_name = f"x_{best_global}"
        name_to_local = {
            branch_cands[i].name: i for i in range(npriocands)
        }

        if best_name not in name_to_local:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        local_idx = name_to_local[best_name]
        self.model.branchVarVal(
            branch_cands[local_idx],
            branch_cand_sols[local_idx],
        )
        return {"result": SCIP_RESULT.BRANCHED}


def solve_default(seed=2026, n=18, p=0.28):
    graph, weights = make_mis_instance(n=n, p=p, seed=seed)
    scip, _, _ = build_scip_mis(graph, weights, name="default")
    scip.optimize()
    return {
        "nodes": int(scip.getNTotalNodes()),
        "time": float(scip.getSolvingTime()),
        "status": str(scip.getStatus()),
    }


def solve_with_gnn(model, seed=2026, n=18, p=0.28):
    graph, weights = make_mis_instance(n=n, p=p, seed=seed)
    scip, _, edges = build_scip_mis(graph, weights, name="gnn_branching")

    rule = GNNBranchRule(graph, weights, edges, model)
    scip.includeBranchrule(
        rule,
        name="gnn_branching",
        desc="GNN branching policy trained from strong branching labels",
        priority=1_000_000,
        maxdepth=-1,
        maxbounddist=1.0,
    )

    scip.optimize()
    return {
        "nodes": int(scip.getNTotalNodes()),
        "time": float(scip.getSolvingTime()),
        "status": str(scip.getStatus()),
    }


def expert_top1_accuracy(model, samples):
    model.eval()
    correct = 0

    with torch.no_grad():
        for sample in samples:
            data = sample.to(device)
            logits = model(data)
            logits = logits.masked_fill(
                ~data["variable"].candidate_mask,
                -1e9,
            )
            pred = int(torch.argmax(logits).item())
            target = int(data["variable"].target_index.item())
            correct += int(pred == target)

    return correct / max(len(samples), 1)


if __name__ == "__main__":
    print("Device:", device)
    print("Strong-branching imitation dataset toplanıyor...")

    train_samples, train_stats = collect_dataset(num_instances=10)
    print("Toplam solver-state örneği:", len(train_samples))
    print("Instance sample sayıları:", [s["samples"] for s in train_stats])

    if not train_samples:
        raise RuntimeError(
            "Branching sample oluşmadı. Graph boyutunu/p değerini artırmayı deneyin."
        )

    model = train_model(train_samples, epochs=40)

    test_samples, test_stats = collect_from_instance(
        seed=999,
        n=16,
        p=0.28,
    )
    accuracy = expert_top1_accuracy(model, test_samples)
    print("Held-out strong-branching top-1 accuracy:", accuracy)
    print("Held-out expert stats:", test_stats)

    default_result = solve_default()
    gnn_result = solve_with_gnn(model)

    print("Default SCIP:", default_result)
    print("GNN branching:", gnn_result)

    print(
        "\nNot: Tek küçük instance sonuçları bilimsel kanıt değildir. "
        "Gerçek benchmark'ta solve time, node count, primal/dual integral, "
        "solved fraction, size generalization ve distribution shift ölçülmelidir."
    )
