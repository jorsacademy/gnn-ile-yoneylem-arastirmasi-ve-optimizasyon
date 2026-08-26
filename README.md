# Graf Sinir Ağları ile Yöneylem Araştırması ve Optimizasyon

> **Güncelleme:** Ağustos 2026  
> Bu repo; Graf Sinir Ağlarının (Graph Neural Networks, GNN) yöneylem araştırması, endüstri mühendisliği ve optimizasyon problemlerinde nasıl kullanılabileceğini açıklayan Türkçe bir rehberdir.

GNN'ler yalnızca düğüm sınıflandırma veya link prediction için kullanılan modeller değildir. Bir optimizasyon problemi doğal olarak bir ağ, çizge, değişken-kısıt ilişkisi, görev-makine ilişkisi veya tesis-müşteri ağı içeriyorsa GNN'ler problem yapısını öğrenmek için güçlü bir araç olabilir.

Bu rehberin temel yaklaşımı şudur:

> **GNN'nin klasik optimizasyon algoritmasının yerini tamamen alması gerekmez. Çoğu gerçek problemde daha iyi yaklaşım, GNN'yi solver'ın aramasını yönlendiren öğrenilmiş bir bileşen olarak kullanmaktır.**

Bu nedenle yalnızca `graph -> GNN -> solution` yaklaşımını değil; GNN + local search, GNN + reinforcement learning, GNN + branch-and-bound, GNN + cutting planes, GNN + CP-SAT/MIP ve GNN + differentiable optimization yaklaşımlarını da ele alıyoruz.

---

## İçindekiler

1. [Neden GNN ve optimizasyon?](#1-neden-gnn-ve-optimizasyon)
2. [Bir optimizasyon problemi nasıl grafa çevrilir?](#2-bir-optimizasyon-problemi-nasıl-grafa-çevrilir)
3. [Hangi GNN türü hangi problemde kullanılabilir?](#3-hangi-gnn-türü-hangi-problemde-kullanılabilir)
4. [GNN optimizasyon sürecinde hangi rolü oynar?](#4-gnn-optimizasyon-sürecinde-hangi-rolü-oynar)
5. [Yöneylem araştırması problem sınıfları](#5-yöneylem-araştırması-problem-sınıfları)
6. [Kütüphaneler ve framework'ler](#6-kütüphaneler-ve-frameworkler)
7. [Önerilen teknoloji yığınları](#7-önerilen-teknoloji-yığınları)
8. [Deneyler nasıl değerlendirilmelidir?](#8-deneyler-nasıl-değerlendirilmelidir)
9. [Başlangıç çalışma planı](#9-başlangıç-çalışma-planı)
10. [Repo örneği](#10-repo-örneği)
11. [Kaynaklar](#11-kaynaklar)

---

# 1. Neden GNN ve optimizasyon?

Birçok yöneylem araştırması problemi zaten graph yapısına sahiptir:

- TSP ve VRP: şehirler/müşteriler düğüm, yollar kenar.
- Şebeke tasarımı: tesisler, depolar, müşteriler veya istasyonlar düğüm.
- Üretim çizelgeleme: operasyonlar düğüm, öncelik ve makine çatışmaları kenar.
- MILP: değişkenler ve kısıtlar iki farklı düğüm tipi oluşturabilir.
- Set Cover / Set Packing: kümeler ve elemanlar bipartite veya hypergraph olarak modellenebilir.
- SAT/CSP: değişkenler ve clause/kısıtlar bipartite graph oluşturabilir.
- Tedarik zinciri: tedarikçi, fabrika, depo, dağıtım merkezi ve müşteri katmanları doğal bir graph oluşturur.
- Enerji ve ulaşım: fiziksel şebeke zaten graph'tır.

Klasik bir optimizasyon modeli

```text
min f(x)

s.t.
    g_i(x) <= 0
    h_j(x)  = 0
    x in X
```

şeklindedir. GNN burada doğrudan `x*` üretmek zorunda değildir. Şunlardan yalnızca birini öğrenmesi bile çok değerli olabilir:

- hangi değişken üzerinde branch edileceği,
- hangi cut'ın ekleneceği,
- hangi düğüm veya kenarın seçileceği,
- hangi local-search hareketinin denenmesi gerektiği,
- hangi başlangıç çözümünün solver'a verilmesi gerektiği,
- hangi değişkenlerin sabitlenebileceği,
- hangi komşuluğun araştırılacağı,
- hangi görev/araç/makine kararının sıradaki aksiyon olacağı.

Bu bakış açısı **learning-augmented optimization** veya **learning to optimize** yaklaşımına yakındır.

---

# 2. Bir optimizasyon problemi nasıl grafa çevrilir?

GNN başarısının en kritik kısmı modelden önce gelir: **graph representation**.

## 2.1 TSP / VRP

```text
Düğüm   = şehir / müşteri / depo
Kenar   = gidilebilir bağlantı
Node feature = koordinat, talep, zaman penceresi, servis süresi
Edge feature = mesafe, süre, maliyet, trafik
```

TSP'de graph çoğu zaman complete graph'tır. Büyük örneklerde k-nearest-neighbor graph gibi seyrekleştirme yöntemleri kullanılabilir.

## 2.2 Job Shop Scheduling

```text
Düğüm = operasyon
Kenar tipi 1 = iş içi öncelik ilişkisi
Kenar tipi 2 = aynı makineyi isteyen operasyonlar arasındaki çatışma
Node feature = işlem süresi, iş ID, makine ID, hazır olma zamanı
```

Bu yapı çoğu zaman **heterogeneous / disjunctive graph** yaklaşımıyla modellenir.

## 2.3 MILP için variable-constraint bipartite graph

MILP:

```text
min c^T x
s.t. A x <= b
     x_i integer  (bazı i'ler için)
```

şu şekilde grafa çevrilebilir:

```text
Düğüm tipi 1 = değişkenler
Düğüm tipi 2 = kısıtlar
Kenar        = A_ij != 0 ise variable j <-> constraint i
Kenar özelliği = A_ij katsayısı
```

Değişken özellikleri:

- amaç fonksiyonu katsayısı,
- LP relaxation değeri,
- reduced cost,
- alt/üst sınır,
- integer/binary bilgisi,
- pseudo-cost.

Kısıt özellikleri:

- RHS,
- slack,
- dual değer,
- constraint türü.

Bu representation, learned branching ve learned primal heuristics çalışmalarında çok kullanışlıdır.

## 2.4 Hypergraph gerektiren problemler

Bir kısıt aynı anda çok sayıda değişkeni bağlıyorsa ikili kenarlar ilişkileri gereğinden fazla parçalayabilir.

Örneğin:

```text
x1 + x7 + x13 + x20 <= 2
```

bir hyperedge olarak düşünülebilir.

Hypergraph yaklaşımı özellikle:

- Set Cover,
- Set Packing,
- SAT/CSP,
- bazı çizelgeleme problemleri,
- kaynak paylaşımı,
- çoklu etkileşim içeren network design

için uygundur.

---

# 3. Hangi GNN türü hangi problemde kullanılabilir?

Tek bir "en iyi GNN" yoktur. Model, graph yapısına ve karar mekanizmasına göre seçilmelidir.

| GNN ailesi | Güçlü tarafı | OR / optimizasyon kullanımı | Dikkat edilmesi gereken |
|---|---|---|---|
| GCN | Basit ve hızlı message passing | Max-Cut, MIS, graph-level heuristics | Homophily varsayımı problem yaratabilir |
| GraphSAGE | Neighbor sampling ve inductive öğrenme | Büyük graphlar, network problems | Aggregator seçimi önemli |
| GAT | Öğrenilmiş attention ağırlıkları | Routing, scheduling, kaynak atama | Büyük graphlarda attention maliyeti |
| GIN | Güçlü yapısal ayırt etme kapasitesi | Combinatorial graph problems | Global bağımlılıklar yine sınırlı olabilir |
| MPNN | Edge feature kullanımında esnek | Mesafe/maliyet/kapasite içeren ağlar | Oversquashing |
| Heterogeneous GNN | Birden fazla node/edge tipi | MILP, JSSP, supply chain | Veri modelleme daha karmaşık |
| Bipartite GNN | İki düğüm tipi arasındaki ilişki | Variable-constraint, clause-variable | Problem yapısına özel tasarım gerekir |
| Hypergraph NN | Higher-order etkileşim | Set Cover, CSP, çoklu kaynak ilişkileri | Daha az standart tooling |
| Graph Transformer | Global attention | TSP/VRP, assignment, global scheduling | O(n²) dikkat maliyeti oluşabilir |
| Spectral GNN | Laplacian/spectral yapı | Max-Cut, partitioning, power networks | Graph boyutuna ve spektruma duyarlılık |
| Higher-order / k-GNN | Standart MPNN'den daha yüksek expressivity | Symmetry ve zor graph ayrımları | Hesaplama maliyeti hızla artar |
| Equivariant / Geometric GNN | Geometrik simetrileri korur | Robotik, 3B yerleşim, fiziksel ağlar | Problem geometrik değilse gereksiz olabilir |
| Temporal / Dynamic GNN | Zamanla değişen graph | Dynamic VRP, trafik, online scheduling | State tanımı kritik |
| Heterophily-aware GNN | Komşuların farklı sınıflarda olması durumunu işler | Max-Cut gibi "komşular farklı olsun" problemleri | Standart GCN varsayımından farklı tasarım |

## 3.1 Standart message-passing modeli

Çoğu GNN kabaca şu yapıya indirgenebilir:

```text
message(u -> v) = psi(h_u, h_v, e_uv)

aggregate(v) = AGG({message(u -> v) : u in N(v)})

h_v_new = phi(h_v, aggregate(v))
```

Optimizasyonda önemli problem şudur: iyi bir karar bazen graph'ın çok uzaktaki bölgelerine bağlıdır. Çok derin message passing ise **oversmoothing** ve **oversquashing** sorunları doğurabilir.

Bu nedenle global ilişkilerin önemli olduğu problemlerde Graph Transformer veya GNN + global feature kombinasyonları düşünülebilir.

---

# 4. GNN optimizasyon sürecinde hangi rolü oynar?

## 4.1 Doğrudan çözüm üretme

```text
Graph -> GNN -> çözüm
```

Örneğin her düğüm için

```text
p_i = P(x_i = 1)
```

üretip threshold/rounding ile binary çözüm elde edilebilir.

Avantaj: inference çok hızlı olabilir.  
Dezavantaj: feasibility ve optimality garantisi yoktur.

---

## 4.2 Heatmap / skor üretme

GNN doğrudan çözüm yerine

```text
p_ij = P((i,j) kenarı çözümde)
```

veya

```text
score(v)
```

üretir.

Ardından:

- greedy,
- beam search,
- 2-opt / 3-opt,
- local search,
- MCTS,
- classical heuristic

kullanılır.

Bu yaklaşım çoğu zaman daha güvenlidir.

---

## 4.3 GNN + Reinforcement Learning

```text
state graph -> GNN encoder -> policy -> action
```

Örnek aksiyonlar:

- sıradaki müşteriyi seç,
- sıradaki operasyonu çizelgele,
- hangi aracı ata,
- hangi local-search move'u uygula.

Reward genellikle objective ile ilişkilidir:

```text
reward = - toplam maliyet
```

veya

```text
reward = - makespan
```

Routing ve dynamic scheduling için çok doğal bir yaklaşımdır.

---

## 4.4 GNN + Local Search

GNN'nin görevi çözümü üretmek yerine hangi komşuluğun daha umut verici olduğunu tahmin etmek olabilir.

```text
mevcut çözüm
    -> GNN
    -> move / neighborhood score
    -> local search
```

Örneğin VRP'de:

- swap,
- relocate,
- 2-opt,
- 3-opt,
- ruin-and-recreate

hareketleri arasından seçim yapılabilir.

---

## 4.5 GNN + Branch-and-Bound

MILP'de GNN şu soruyu öğrenebilir:

> Hangi değişken üzerinde branch etmeliyim?

```text
MILP
 -> variable-constraint bipartite graph
 -> GNN
 -> her değişken için branching score
 -> SCIP / başka MIP solver
```

Strong branching'den imitation learning etiketi üretmek yaygın bir araştırma yaklaşımıdır.

GNN burada solver'ın doğruluğunu değiştirmez; **arama sırasını** değiştirmeye çalışır.

---

## 4.6 GNN + Cutting Planes

GNN şu kararı öğrenebilir:

```text
hangi geçerli cut'lar eklenmeli?
```

Model yalnızca solver'ın ürettiği valid cut adayları arasından seçim yaparsa correctness mekanizması klasik optimizer'da kalır.

---

## 4.7 Warm start / primal heuristic

GNN yaklaşık bir çözüm tahmini üretir:

```text
p_i = P(x_i = 1)
```

Bu çözüm:

- MIP start,
- feasibility repair,
- local branching,
- neighborhood search

için kullanılabilir.

Gerçek endüstriyel kullanım için en mantıklı GNN rollerinden biridir.

---

## 4.8 Differentiable optimization

Başka bir paradigma:

```text
GNN -> problem parametreleri -> differentiable optimization layer -> karar
```

Örneğin talep/maliyet parametrelerini graph üzerinden tahmin edip ardından convex optimization katmanı çalıştırılabilir.

Bu yaklaşım **predict-then-optimize** problemlerinde kullanışlıdır.

---

# 5. Yöneylem araştırması problem sınıfları

## 5.1 TSP / VRP / CVRP / VRPTW

Önerilen modeller:

- attention tabanlı GNN,
- Graph Transformer,
- MPNN,
- RL4CO modelleri.

Önerilen çözüm stratejileri:

```text
constructive policy
GNN heatmap + search
GNN + local search
GNN + RL
```

---

## 5.2 Job Shop / Flexible Job Shop / Flow Shop

Representation:

```text
operation nodes
precedence edges
machine-conflict edges
```

Uygun modeller:

- heterogeneous GNN,
- GAT,
- Graph Transformer,
- temporal GNN (dinamik çizelgeleme).

Amaçlar:

- makespan,
- tardiness,
- setup cost,
- enerji maliyeti.

---

## 5.3 Facility Location ve Supply Chain Network Design

Graph:

```text
supplier -> factory -> warehouse -> customer
```

GNN şu işlerde kullanılabilir:

- tesis açma olasılığı tahmini,
- candidate arc pruning,
- warm start,
- scenario embedding,
- decomposition yönlendirme.

Son kararı Pyomo / Gurobi / SCIP gibi solver'lara bırakmak çoğu zaman daha doğru olur.

---

## 5.4 Max-Cut

Max-Cut özellikle ilginçtir çünkü komşu düğümlerin **farklı** kümelerde olması istenir. Bu nedenle standart homophily odaklı GNN varsayımları ideal değildir.

Kullanılabilecek yaklaşımlar:

- heterophily-aware GNN,
- spectral GNN,
- GNN + differentiable surrogate,
- QUBO + GNN.

Bu repodaki ilk notebook bu problemi kullanır.

---

## 5.5 MIS / Maximum Clique / Vertex Cover / Coloring

GNN her düğüm için bir seçim skoru üretebilir:

```text
score(v)
```

Sonrasında:

```text
GNN score -> greedy construction -> feasibility repair -> local search
```

pipeline'ı kullanılabilir.

---

## 5.6 MILP

MILP tarafında en güçlü kullanım alanları:

- learned branching,
- learned node selection,
- learned cut selection,
- learned primal heuristics,
- variable fixing,
- warm start,
- presolve kararları.

Burada GNN'yi çoğu zaman **solver replacement** değil **solver guidance** olarak düşünmek gerekir.

---

## 5.7 SAT / CSP

Representation:

```text
variable nodes <-> clause / constraint nodes
```

GNN:

- variable score,
- phase prediction,
- clause importance,
- branching guidance

üretebilir.

Yine klasik SAT/CDCL veya CP solver correctness mekanizmasını koruyabilir.

---

## 5.8 Multi-echelon inventory ve üretim ağları

Tek ürünlü klasik stok modeli doğrudan graph problemi olmayabilir. Ancak çok kademeli sistemlerde

```text
supplier -> plant -> DC -> retailer
```

ilişkileri doğal graph'tır.

Temporal GNN + RL kombinasyonu şu problemlerde kullanılabilir:

- replenishment,
- allocation,
- transshipment,
- disruption response.

---

# 6. Kütüphaneler ve framework'ler

## 6.1 PyTorch

Derin öğrenme altyapısı için temel tercih.

```bash
pip install torch
```

Kendi GNN layer'ınızı yazmak, custom loss tanımlamak veya RL/solver entegrasyonu yapmak için temel framework'tür.

---

## 6.2 PyTorch Geometric (PyG) — genel GNN çalışmaları için ilk tercih

Resmi dokümantasyon: https://pytorch-geometric.readthedocs.io/

PyG;

- GCN,
- GraphSAGE,
- GAT,
- GIN,
- heterogeneous graph,
- sampling,
- mini-batching,
- büyük graph loader'ları,
- `torch.compile`,
- multi-GPU senaryoları

için güçlü bir ekosistem sunar.

Kurulumun temel hali:

```bash
pip install torch_geometric
```

Ağustos 2026 itibarıyla yeni genel-purpose GNN araştırmaları için varsayılan tercihimiz **PyTorch + PyG**.

---

## 6.3 NetworkX — prototipleme ve graph algoritmaları

https://networkx.org/

GNN framework değildir. Ancak küçük graph üretmek, graph istatistikleri hesaplamak, classical baseline oluşturmak ve sonuçları doğrulamak için çok kullanışlıdır.

```bash
pip install networkx
```

---

## 6.4 RL4CO — Neural Combinatorial Optimization

https://github.com/ai4co/rl4co

Routing ve scheduling başta olmak üzere neural combinatorial optimization araştırmaları için güçlü bir framework'tür.

Desteklediği ana paradigma sınıfları:

- autoregressive constructive,
- non-autoregressive constructive,
- improvement policies,
- transductive yöntemler.

TSP/CVRP/JSSP gibi problem sınıflarında environment ve eğitim altyapısını sıfırdan yazmak yerine başlangıç noktası olabilir.

```bash
pip install rl4co
```

---

## 6.5 SCIP + PySCIPOpt — learned MIP solver için

SCIP: https://www.scipopt.org/  
PySCIPOpt: https://pyscipopt.readthedocs.io/

MILP tarafında GNN ile solver kontrol etmek istiyorsanız en önemli araçlardan biridir.

Özelleştirilebilen solver bileşenleri arasında:

- branching rule,
- heuristics,
- separators,
- node selection,
- cut selection

gibi mekanizmalar bulunur.

Önerilen araştırma stack'i:

```text
PyTorch
+ PyTorch Geometric
+ PySCIPOpt
+ SCIP
```

---

## 6.6 Ecole — SCIP'i ML/RL ortamı gibi kullanmak

https://doc.ecole.ai/

Ecole, SCIP'i reinforcement learning / machine learning deneylerine uygun bir environment biçiminde kullanmayı amaçlayan araştırma aracıdır.

Özellikle learned branching çalışmalarını anlamak için değerlidir. Ancak yeni proje başlatırken paket/SCIP sürüm uyumluluğu ve projenin güncel bakım durumunu ayrıca kontrol etmek gerekir.

---

## 6.7 OR-Tools

https://developers.google.com/optimization

Google OR-Tools klasik optimizasyon baseline'ı ve hibrit solver tasarımı için çok değerlidir.

Özellikle:

- CP-SAT,
- scheduling,
- routing,
- assignment,
- packing,
- network flow

problemlerinde kullanılır.

GNN ile örneğin candidate kararları veya warm-start benzeri rehber bilgiler üretip klasik OR-Tools çözümleriyle kıyaslama yapılabilir.

---

## 6.8 Pyomo

https://pyomo.readthedocs.io/

Matematiksel programlama modellerini Python'da ifade etmek için güçlü bir modeling layer'dır.

GNN'nin tahmin ettiği parametreleri veya candidate decision'ları son optimizasyon modeline vermek için uygundur.

Örnek:

```text
GNN -> candidate facilities -> Pyomo MILP -> exact/near-exact solve
```

---

## 6.9 Gurobi

https://www.gurobi.com/

Ticari optimizasyon solver'ı. Akademik ve endüstriyel benchmarklarda önemlidir.

GNN ile:

- MIP start,
- variable hint,
- candidate pruning,
- callback tabanlı araştırmalar

yapılabilir; ancak entegrasyon tasarımı solver API'sinin izin verdiği müdahale noktalarına bağlıdır.

---

## 6.10 DGL

https://www.dgl.ai/

Deep Graph Library hâlâ kullanılabilir bir GNN framework'üdür. PyTorch ekosistemiyle çalışabilir ve distributed graph learning yetenekleri vardır.

Bununla birlikte NVIDIA'nın cuGraph GNN entegrasyonunda yönelim artık PyG tarafındadır. Yeni NVIDIA GPU merkezli large-scale projelerde bu nedenle PyG ekosistemi daha doğal tercih olabilir.

---

## 6.11 cuGraph-PyG — büyük ölçekli GPU graph learning

https://docs.nvidia.com/cugraph/

Ağustos 2026 itibarıyla NVIDIA'nın GNN tarafındaki ana graph entegrasyonu `cuGraph-PyG` yönündedir.

Kullanım alanları:

- GPU graph sampling,
- heterogeneous sampling,
- distributed graph storage,
- multi-GPU GNN training,
- çok büyük graphlar.

Önerilen stack:

```text
PyTorch
+ PyTorch Geometric
+ cuGraph-PyG
+ WholeGraph (gerektiğinde)
```

---

# 7. Önerilen teknoloji yığınları

## Genel GNN + optimizasyon araştırması

```text
Python
PyTorch
PyTorch Geometric
NetworkX
NumPy / SciPy
Jupyter
```

## TSP / CVRP / Neural Combinatorial Optimization

```text
PyTorch
PyTorch Geometric
RL4CO
OR-Tools veya PyVRP / klasik heuristic baseline
```

## MILP + learned branching / cut / heuristic

```text
PyTorch
PyTorch Geometric
PySCIPOpt
SCIP
Pyomo (modelleme gerekiyorsa)
```

## Scheduling

```text
PyTorch Geometric
RL4CO veya custom RL
OR-Tools CP-SAT
Pyomo / Gurobi
```

## Large-scale graph optimization

```text
PyTorch
PyTorch Geometric
cuGraph-PyG
WholeGraph
```

## Production ortamı için genel tercih

```text
GNN
   |
   v
arama skoru / warm start / candidate reduction
   |
   v
klasik solver
   |
   v
feasible + doğrulanmış çözüm
```

Bu yapı genellikle

```text
GNN -> final solution
```

yaklaşımından daha güvenlidir.

---

# 8. Deneyler nasıl değerlendirilmelidir?

Bir GNN optimizasyon çalışmasını yalnız neural-network loss veya accuracy ile değerlendirmek doğru değildir.

## 8.1 Objective value

Minimizasyon probleminde:

```text
f(x_GNN)
```

ölçülür.

## 8.2 Optimality gap

Optimal veya güçlü bir referans çözüm biliniyorsa:

```text
gap = (f_GNN - f_best) / |f_best|
```

probleme uygun işaret konvansiyonu ile hesaplanır.

## 8.3 Feasibility rate

```text
uygulanabilir çözüm sayısı / toplam instance
```

GNN çözüm üretip sık sık constraint ihlal ediyorsa objective iyi görünse bile yöntem pratik değildir.

## 8.4 Time-to-solution

En önemli metriklerden biri:

```text
GNN inference
+ repair
+ solver
+ local search
```

toplam süresidir.

GNN klasik heuristic'ten daha iyi karar veriyor olabilir ama inference overhead'i nedeniyle toplam solver süresini artırabilir.

## 8.5 Size generalization

Örneğin:

```text
train: 50-100 node
validation: 100 node
test: 200 / 500 / 1000 node
```

GNN'nin graph boyutuna genellemesi ayrıca ölçülmelidir.

## 8.6 Distribution shift

Sadece aynı graph generator ile train/test yapılmamalıdır.

Örneğin:

```text
train: Erdos-Renyi
test: Barabasi-Albert / geometric / gerçek veri
```

ile robustness ölçülebilir.

## 8.7 Güçlü classical baseline

GNN'yi yalnız random veya zayıf greedy baseline ile kıyaslamak yeterli değildir.

Problem türüne göre:

- OR-Tools,
- SCIP,
- Gurobi,
- CP-SAT,
- iyi local search,
- metaheuristic,
- problem-özel heuristic

ile karşılaştırma yapılmalıdır.

---

# 9. Başlangıç çalışma planı

Bu alanı öğrenmek için önerilen sıra:

### Aşama 1 — Graph temelleri

- adjacency matrix/list,
- degree,
- shortest path,
- Laplacian,
- message passing.

### Aşama 2 — PyTorch Geometric

- `Data`,
- `edge_index`,
- `GCNConv`,
- `SAGEConv`,
- `GATConv`,
- batching.

### Aşama 3 — Küçük combinatorial optimization

- Max-Cut,
- MIS,
- Vertex Cover.

Önce label-free differentiable objective denenebilir.

### Aşama 4 — Hybrid solver

```text
GNN score -> greedy / repair / local search
```

### Aşama 5 — MILP representation

Variable-constraint bipartite graph oluşturun.

### Aşama 6 — Solver-in-the-loop

```text
PyG + PySCIPOpt
```

ile branching veya primal heuristic deneyin.

### Aşama 7 — RL tabanlı CO

```text
RL4CO
```

ile TSP/CVRP/JSSP deneyleri yapın.

---

# 10. Repo örneği

`notebooks/01_maxcut_gnn.ipynb` dosyasında küçük bir **Max-Cut + PyTorch Geometric** örneği bulunmaktadır.

Notebook'un amacı en iyi Max-Cut solver'ını yazmak değildir. Amaç şu pipeline'ı açık biçimde göstermektir:

```text
Graph
 -> GCN
 -> node probabilities
 -> differentiable Max-Cut surrogate
 -> training
 -> rounding
 -> discrete solution
 -> optimality-gap kontrolü
```

Bu örnek daha sonra şu şekilde geliştirilebilir:

```text
GCN
 -> GraphSAGE / GIN / GAT
 -> randomized rounding
 -> local search
 -> solver warm-start
```

---

# 11. Kaynaklar

## Resmi dokümantasyon

- PyTorch: https://pytorch.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- NetworkX: https://networkx.org/
- RL4CO: https://github.com/ai4co/rl4co
- SCIP: https://www.scipopt.org/
- PySCIPOpt: https://pyscipopt.readthedocs.io/
- Ecole: https://doc.ecole.ai/
- OR-Tools: https://developers.google.com/optimization
- Pyomo: https://pyomo.readthedocs.io/
- DGL: https://www.dgl.ai/
- NVIDIA cuGraph-PyG: https://docs.nvidia.com/cugraph/

## Başlangıç için önemli araştırma başlıkları

Literatür tararken şu anahtar kelimeler yararlıdır:

```text
neural combinatorial optimization
learning to branch
learning to cut
learning augmented optimization
GNN for mixed integer programming
GNN for vehicle routing
GNN job shop scheduling
GNN Max-Cut
neural algorithmic reasoning
graph reinforcement learning optimization
differentiable optimization graph neural networks
```

Klasik başlangıç çalışmaları arasında **Learning Combinatorial Optimization Algorithms over Graphs**, **NeuroSAT**, attention tabanlı routing çalışmaları ve variable-constraint bipartite GNN ile learned branching literatürü incelenebilir. Daha yeni çalışmalar için güncel konferans ve dergilerde tarih filtresiyle arama yapılması önerilir.

---

# Ana fikir

Bir endüstri mühendisi veya yöneylem araştırmacısı için en önemli soru:

> **"Bu problemi GNN ile tamamen çözebilir miyim?"**

yerine çoğu zaman şudur:

> **"Optimizasyon algoritmasının hangi pahalı kararını GNN ile daha hızlı veya daha iyi tahmin edebilirim?"**

Bu karar;

- branch variable,
- cut,
- neighborhood,
- route candidate,
- machine assignment,
- warm start,
- variable fixing,
- search priority

olabilir.

GNN'lerin yöneylem araştırmasındaki en güçlü kullanım alanı, klasik optimizasyon teorisini ortadan kaldırmak değil; **öğrenilmiş heuristics ile klasik algoritmaları güçlendirmektir.**
