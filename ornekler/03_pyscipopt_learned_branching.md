# PySCIPOpt ile GNN Tabanlı Learned Branching

Bu örnek, `03_pyscipopt_learned_branching.py` dosyasının ne yaptığını açıklar.

## Amaç

GNN'nin bir MILP solver'ın yerini alması yerine branch-and-bound içindeki pahalı **branching variable selection** kararını öğrenmesini göstermek.

Akış:

```text
MILP / solver state
      |
      v
variable-constraint bipartite graph
      |
      v
strong branching expert
      |
      v
imitation-learning dataset
      |
      v
bipartite GNN
      |
      v
branching score
      |
      v
SCIP branchVarVal(...)
```

Bu yaklaşımın temel literatür örneği:

- Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, Andrea Lodi — **Exact Combinatorial Optimization with Graph Convolutional Neural Networks**, NeurIPS 2019  
  https://arxiv.org/abs/1906.01629

## Neden strong branching?

Strong branching, aday değişkenler üzerinde geçici olarak aşağı/yukarı branch ederek oluşacak LP dual-bound gelişimini tahmin eder. Genellikle kaliteli branching kararı üretir fakat pahalıdır.

Learned branching fikri:

```text
pahalı strong branching
        -> eğitim etiketi
        -> ucuz GNN inference
```

şeklindedir.

## PySCIPOpt'ta kullanılan gerçek API

Örnekte şu PySCIPOpt mekanizmaları kullanılır:

```python
model.getLPBranchCands()
model.startStrongbranch()
model.getVarStrongbranch(...)
model.endStrongbranch()
model.getBranchScoreMultiple(...)
model.branchVarVal(...)
model.includeBranchrule(...)
```

Resmi branching dokümantasyonu:  
https://pyscipopt.readthedocs.io/en/latest/tutorials/branchrule.html

## Eğitim problemi

Örnek, ağırlıklı Maximum Independent Set kullanır:

\[
\max \sum_{v\in V} w_v x_v
\]

\[
x_u+x_v\le 1 \qquad \forall (u,v)\in E
\]

\[
x_v\in\{0,1\}.
\]

SCIP içinde eşdeğer minimizasyon formu kullanılır:

\[
\min -\sum_v w_vx_v.
\]

Her graph kenarı bir MILP kısıtına dönüşür. Böylece doğal variable-constraint bipartite representation elde edilir.

## GNN feature'ları

Variable node:

- objective coefficient,
- mevcut LP değeri,
- fractionality,
- local lower bound,
- local upper bound,
- graph degree.

Constraint node:

- RHS,
- mevcut LP slack,
- constraint degree.

Edge:

- doğrusal katsayı.

Genel MILP araştırmasında buna reduced cost, dual, pseudo-cost, basis status, incumbent bilgisi ve değişken/kısıt türleri eklenebilir.

## Model

Message passing:

```text
variables -> constraints -> variables
```

GNN sonunda her değişken için bir branching logit üretir. Yalnız mevcut LP branching adayları maskelenmeden bırakılır.

Training objective, strong branching'in seçtiği en iyi değişkeni taklit eden masked cross-entropy'dir.

## Çalıştırma

Repo kökünde:

```bash
pip install -r requirements.txt
python ornekler/03_pyscipopt_learned_branching.py
```

Strong branching veri toplama aşaması normal branching'den pahalıdır; bu beklenen davranıştır.

## Sonuçları nasıl yorumlamamalıyız?

Tek bir küçük örnekte GNN'nin default SCIP'ten hızlı olması bilimsel kanıt değildir.

Gerçek benchmark en az şunları ölçmelidir:

- wall-clock solve time,
- branch-and-bound node count,
- solved fraction,
- primal integral,
- dual integral,
- time-to-first-feasible-solution,
- size generalization,
- distribution shift.

Karşılaştırmalar:

```text
default SCIP
most fractional
pseudo-cost
strong branching
GNN learned branching
```

şeklinde yapılmalıdır.

## Ana fikir

Bu örnek GNN'nin optimizasyon teorisini ortadan kaldırdığı iddiasında değildir.

GNN yalnızca şu pahalı kararı öğrenir:

> **Bu branch-and-bound düğümünde hangi integer variable üzerinde branch etmeliyim?**

Feasibility, bounding, node processing ve optimality proof yine SCIP tarafından yürütülür.
