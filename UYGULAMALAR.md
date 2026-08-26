# Uygulamalı Örnekler

Bu repo yalnız GNN mimarilerini listelemek için değil, bu mimarilerin yöneylem araştırması ve optimizasyon içinde hangi rolü oynayabileceğini somutlaştırmak için hazırlanmıştır.

## 01 — GNN ile Max-Cut

Dosya: `notebooks/01_maxcut_gnn.ipynb`

Amaç:

- graph optimizasyonunu GNN ile temsil etmek,
- etiketsiz/diferansiyellenebilir objective kullanmak,
- continuous node skorlarını discrete çözüme yuvarlamak,
- brute-force optimum ile optimality gap kontrolü yapmak.

Temel fikir:

```text
Graph -> GCN -> node probabilities -> differentiable Max-Cut objective -> rounding
```

## 02 — MILP'yi Variable–Constraint Bipartite Grafa Dönüştürme

Dosya: `notebooks/02_milp_variable_constraint_gnn.ipynb`

Amaç:

- MILP katsayı matrisi `A` üzerinden bipartite graph kurmak,
- değişken ve kısıt düğümlerini ayrı temsil etmek,
- `A_ij` katsayılarını edge feature olarak kullanmak,
- warm-start / primal heuristic fikrini göstermek,
- branching için GNN skoru + LP fractionality ilişkisini açıklamak.

Temel fikir:

```text
MILP -> variable/constraint graph -> bipartite GNN -> variable score
```

## 03 — PySCIPOpt ile Learned Branching

Dosya: `ornekler/03_pyscipopt_learned_branching.py`

Amaç:

- SCIP branch-and-bound döngüsüne gerçekten bağlanmak,
- strong branching'i expert olarak kullanmak,
- solver state'lerinden imitation-learning etiketi toplamak,
- GNN branching rule'u tekrar SCIP içine yerleştirmek.

Temel fikir:

```text
SCIP state
 -> variable-constraint bipartite graph
 -> strong-branching expert labels
 -> GNN imitation learning
 -> custom SCIP Branchrule
```

Bu örnek, Gasse vd. (NeurIPS 2019) çizgisindeki learned branching yaklaşımının eğitim amaçlı küçük bir uygulamasıdır; ilgili çalışmanın birebir reprodüksiyonu değildir.

## 04 — R-GCN + MILP ile Tedarik Zincirinde Aday Hat Budama

Dosya: `notebooks/04_supply_chain_rgcn.ipynb`

Amaç:

- multi-echelon tedarik zincirini multi-relational graph olarak modellemek,
- `supplier -> plant -> warehouse -> customer` ilişkilerini R-GCN relation tipleri olarak kullanmak,
- tam MILP optimumunda kullanılan hatları eğitim etiketi yapmak,
- GNN ile candidate arc skoru üretmek,
- düşük skorlu arc'ları kontrollü biçimde budamak,
- küçültülmüş MILP'yi tekrar çözmek,
- objective gap, feasibility, retained-arc oranı ve solve time ölçmek,
- GNN'yi basit cost-only heuristic ile karşılaştırmak.

Temel fikir:

```text
full supply-chain MILP
 -> optimal arc labels
 -> R-GCN edge scorer
 -> candidate arc pruning
 -> adaptive feasibility fallback
 -> smaller MILP
```

Buradaki kritik güvenlik prensibi:

> GNN karar uzayını daraltmayı önerir; uygulanabilirlik ve nihai karar yine optimizasyon solver'ı tarafından doğrulanır.

## 05 — Directed GNN + Min-Cost Flow

Dosya: `notebooks/05_directed_gnn_network_flow.ipynb`

Amaç:

- yönlü min-cost-flow ağları üretmek,
- full LP optimumunda kullanılan arc'ları eğitim etiketi yapmak,
- graph'ı simetrize eden bir undirected GNN ile incoming/outgoing mesajları ayrı işleyen directed GNN'yi karşılaştırmak,
- optimal-arc recall ölçmek,
- GNN skorlarıyla candidate arc pruning yapmak,
- infeasible pruning durumunda adaptive fallback uygulamak,
- objective gap, retained-arc oranı ve çözüm süresini ölçmek,
- cost-only heuristic ile karşılaştırmak.

Temel fikir:

```text
directed min-cost-flow network
 -> full LP
 -> optimal arc labels
 -> undirected GNN vs directed GNN
 -> arc pruning
 -> feasibility fallback
 -> smaller LP
```

Bu örnekte yön bilgisi problem semantiğinin kendisidir: `u -> v` ve `v -> u` aynı karar değildir. Bu nedenle directed message passing yalnız mimari tercihi değil, graph temsilinin doğruluğuyla ilgili bir konudur.

## Önerilen öğrenme sırası

```text
01 Max-Cut
   ↓
02 MILP bipartite representation
   ↓
03 solver-in-the-loop learned branching
   ↓
04 relational GNN + hybrid network-design optimization
   ↓
05 directed GNN + network-flow candidate pruning
```

Bu beş örnek birlikte GNN kullanımının beş farklı biçimini gösterir:

1. doğrudan differentiable combinatorial objective,
2. optimizasyon modelini graph olarak temsil etme,
3. solver'ın arama kararını öğrenme,
4. relational GNN ile karar uzayını küçültüp klasik MILP ile doğrulama,
5. yönlü network yapısını koruyarak candidate arc filtering yapma.
