# GNN Taksonomisi: Yöneylem Araştırması ve Endüstri Mühendisliği Açısından

> **Güncelleme:** Ağustos 2026  
> Bu belge, ana rehberdeki GNN listesini genişletir. Amaç her yeni mimari adını toplamak değil; yöneylem araştırması (OR), optimizasyon ve endüstri mühendisliği problemlerinde gerçekten anlamlı olan aileleri ayırmaktır.

## 1. Önce ayrım: GNN ailesi mi, graph yapısı mı, teknik mi?

Literatürde üç farklı şey bazen aynı listede karışır:

1. **Mimari aile:** GCN, GAT, GIN, R-GCN, GGNN gibi.
2. **Problem/graph yapısı:** directed graph, heterogeneous graph, hypergraph, signed graph gibi.
3. **Teknik:** graph rewiring, positional encoding, diffusion, sampling gibi.

Bu repo bu üç grubu ayrı düşünür. Örneğin `Directed GNN` tek bir layer adı değildir; yön bilgisini koruyan GNN tasarımlarının genel sınıfıdır. `Rewiring` ise tek başına yeni bir GNN ailesi değil, message passing'in bilgi akışını iyileştiren bir tekniktir.

---

## 2. Yerleşik ve OR açısından doğrudan önemli aileler

### 2.1 Directed GNN

Yönlü graph'larda `u -> v` ile `v -> u` aynı ilişki değildir. Bu, OR'da çok yaygındır:

- tedarik zinciri akışları,
- tek yönlü ulaşım ağları,
- precedence ilişkileri,
- project networks,
- network flow,
- üretim rotaları.

Basit bir directed message-passing tasarımı incoming ve outgoing mesajları ayrı toplar:

\[
m_v^{in}=\operatorname{AGG}_{u:(u,v)\in E}\phi_{in}(h_u)
\]

\[
m_v^{out}=\operatorname{AGG}_{u:(v,u)\in E}\phi_{out}(h_u)
\]

ve sonra

\[
h_v'=\psi(h_v,m_v^{in},m_v^{out}).
\]

Bu gerçek bir model sınıfıdır. Örnek kaynak: Tong vd., **Directed Graph Convolutional Network**  
https://arxiv.org/abs/2004.13970

**Repo açısından:** temel/önemli.

### 2.2 Relational GNN / R-GCN

Bir graph'ta farklı ilişki türleri varsa tek ağırlık matrisi yeterli olmayabilir.

Job Shop örneği:

```text
operation --precedes--> operation
operation --uses--> machine
machine --compatible_with--> operation
```

Supply chain örneği:

```text
supplier --supplies--> plant
plant --ships_to--> warehouse
warehouse --serves--> customer
```

R-GCN ilişki türüne göre ayrı dönüşüm kullanır:

\[
h_i^{(l+1)}=\sigma\left(W_0^{(l)}h_i^{(l)}+\sum_{r\in R}\sum_{j\in N_i^r}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}\right).
\]

Kaynak: Schlichtkrull vd., **Modeling Relational Data with Graph Convolutional Networks**  
https://arxiv.org/abs/1703.06103

**Repo açısından:** temel/önemli.

### 2.3 Edge-centric GNN ve Line-Graph yaklaşımı

Bazı OR problemlerinde karar düğümde değil kenardadır:

\[
x_{ij}\in\{0,1\}.
\]

Örnekler:

- TSP/VRP: rota kenarı seçimi,
- network design: arc açma,
- telecommunication routing,
- enerji iletim hattı seçimi,
- transportation arc flow.

Bir yaklaşım doğrudan edge embedding üretmektir:

\[
h_{ij}^{edge}=\phi(h_i,h_j,e_{ij}).
\]

Diğer yaklaşım **line graph** kullanmaktır. Original graph'ın her kenarı line graph'ta bir düğüme dönüşür:

```text
original graph edge -> line graph node
```

Bu dönüşüm edge-level kararları node-level öğrenme problemine çevirebilir.

Kaynak: Cai vd., **Line Graph Neural Networks for Link Prediction**  
https://doi.org/10.1109/TPAMI.2021.3080635

**Repo açısından:** temel/önemli.

### 2.4 Gated / Recurrent GNN (GGNN)

Message passing, iteratif bir state update olarak modellenebilir:

\[
h_v^{t+1}=GRU(h_v^t,m_v^t).
\]

Özellikle:

- constraint propagation,
- learned dynamic programming,
- iterative scheduling,
- SAT/CSP,
- neural algorithmic reasoning

için doğal olabilir.

**Repo açısından:** orta öncelik.

### 2.5 PNA — Principal Neighbourhood Aggregation

PNA tek aggregator yerine mean, max, min, standard deviation gibi birkaç istatistiği birlikte kullanır ve degree bilgisine göre ölçekleme yapabilir.

Degree dağılımının çok heterojen olduğu supply chain, telecommunication ve constraint graph'larında güçlü bir baseline olabilir.

**Repo açısından:** MPNN ailesi altında faydalı.

### 2.6 Signed GNN

Kenarların yalnız bağlantı değil pozitif/negatif ilişki taşıdığı graph'lar için tasarlanır.

OR örnekleri:

- conflict graph,
- complementarity,
- antagonistic assignment ilişkileri,
- bazı partitioning temsilleri.

Max-Cut için signed GNN zorunlu değildir; heterophily-aware veya spectral modeller de kullanılabilir.

**Repo açısından:** opsiyonel.

---

## 3. Önemli ama ayrı GNN ailesi sayılmaması gereken teknikler

### 3.1 Graph rewiring

Message-passing GNN'lerde uzak bilginin dar bağlantılardan geçmesi **oversquashing** yaratabilir. Rewiring, optimizasyon probleminin kendisini değiştirmek zorunda değildir; GNN'nin kullandığı computational graph yeniden bağlanabilir.

```text
uzak bağımlılık
-> daha kısa bilgi yolu
-> daha az oversquashing
```

Büyük MILP constraint graph'ları, routing, supply chain ve power-network problemleri için araştırmaya değerdir.

**Repo açısından:** önemli teknik; ayrı ana GNN ailesi değildir.

### 3.2 Diffusion / propagation

Personalized PageRank, heat diffusion veya benzeri graph diffusion mekanizmaları klasik birkaç-hop message passing'e alternatif bilgi yayılımı sağlayabilir.

**Repo açısından:** ileri teknik.

### 3.3 Positional / structural encoding

Özellikle Graph Transformer ve global combinatorial optimization modellerinde:

- Laplacian eigenvectors,
- shortest-path distance,
- random-walk encodings,
- centrality features

kullanılabilir. Bunlar yeni GNN ailesi değildir; yapısal bilgi sağlar.

---

## 4. İleri/topolojik modeller

Bu sınıflar gerçektir; ancak genel OR uygulayıcısı için henüz default araç değildir.

### 4.1 Simplicial Neural Networks

Graph yalnız node-edge ilişkisi içerirken simplicial complex daha yüksek boyutlu yapıları da temsil eder:

```text
node -> edge -> triangle/face -> higher-order simplex
```

Potansiyel kullanım: fiziksel ağlar, multi-way flows, transportation topology, complex infrastructure systems.

**Repo açısından:** ileri araştırma konusu.

### 4.2 Cell-complex Neural Networks

Simplicial yapıdan daha genel hücresel yapıları temsil eder. Higher-order topology gerektiren fiziksel veya ağ tabanlı sistemlerde anlamlı olabilir.

**Repo açısından:** ileri araştırma konusu.

### 4.3 Sheaf Neural Networks

Farklı node/edge'lerdeki bilgilerin farklı uzaylarda yaşaması ve aralarında uyumluluk dönüşümleri olması fikrini kullanır.

Gerçek bir araştırma alanıdır; fakat klasik TSP, VRP, MILP veya scheduling için başlangıç tercihi değildir.

**Repo açısından:** ileri/niş araştırma konusu.

---

## 5. Generative graph modelleri

Graph Autoencoder, Variational Graph Autoencoder ve graph generative modeller solver ile karıştırılmamalıdır.

Kullanım alanları:

- synthetic optimization instance generation,
- scenario generation,
- candidate solution generation,
- learned neighborhood generation,
- representation pretraining.

**Repo açısından:** yan dal; solver guidance kadar merkezi değil.

---

## 6. OR açısından önerilen taksonomi

```text
GNN
│
├── Temel Message Passing
│   ├── GCN
│   ├── GraphSAGE
│   ├── GAT
│   ├── GIN
│   ├── MPNN
│   └── PNA
│
├── Yapısal / İlişkisel
│   ├── Directed GNN
│   ├── Bipartite GNN
│   ├── Heterogeneous GNN
│   ├── Relational GNN / R-GCN
│   ├── Hypergraph GNN
│   ├── Edge-centric / Line-Graph GNN
│   └── Signed GNN
│
├── Global / Expressivity
│   ├── Graph Transformer
│   ├── Higher-order / k-GNN
│   ├── Spectral GNN
│   └── Heterophily-aware GNN
│
├── Uzay ve Zaman
│   ├── Temporal GNN
│   ├── Dynamic GNN
│   └── Geometric / Equivariant GNN
│
├── Iterative
│   └── Gated / Recurrent GNN
│
└── İleri Topolojik
    ├── Simplicial NN
    ├── Cell-complex NN
    └── Sheaf NN
```

Rewiring, diffusion, positional encoding ve sampling ise mimariyi tamamlayan tekniklerdir.

---

## 7. Problem → yapı → model eşleştirmesi

| OR problemi | En doğal graph yapısı | Öncelikli GNN yaklaşımı |
|---|---|---|
| TSP / VRP | directed veya complete weighted graph | MPNN, Graph Transformer, edge-centric GNN |
| Asimetrik TSP | directed weighted graph | Directed GNN, attention/Transformer |
| Job Shop | multi-relational/disjunctive graph | Heterogeneous GNN, R-GCN, GAT |
| MILP | variable–constraint bipartite graph | Bipartite GNN / MPNN |
| Set Cover | bipartite / hypergraph | Bipartite GNN, Hypergraph GNN |
| SAT / CSP | variable–clause graph | Bipartite GNN, recurrent GNN |
| Supply chain | directed heterogeneous graph | Directed GNN, R-GCN |
| Network design | edge-decision graph | Edge-centric / Line-Graph GNN |
| Max-Cut | heterophilic graph | Heterophily-aware, spectral, signed (opsiyonel) |
| Power networks | physical directed/undirected graph | MPNN, spectral, geometric |
| Dynamic routing | temporal directed graph | Temporal/Dynamic GNN + RL |

---

## 8. "Matematiksel fantezi" filtresi

### Yerleşik ve uygulanabilir

- Directed GNN
- R-GCN
- edge-centric / line-graph GNN
- GGNN
- PNA
- heterogeneous/bipartite GNN
- hypergraph GNN
- Graph Transformer
- temporal/dynamic GNN

Bunlar yayımlanmış ve uygulanabilir model sınıflarıdır.

### Gerçek ama araştırma-ağırlıklı

- graph rewiring,
- simplicial neural networks,
- cell-complex neural networks,
- sheaf neural networks.

Bunlar uydurma değildir; fakat genel OR practitioner için henüz default araç değildir.

### Kaçınılması gereken hata

Bir modeli yalnızca daha yeni veya daha matematiksel olduğu için seçmek.

Doğru soru:

> **Problem yapısı bu modelin inductive bias'ını gerçekten gerektiriyor mu?**

Örneğin klasik bir küçük/orta MILP için sheaf neural network kullanmak çoğu durumda gereksizdir. Variable–constraint bipartite MPNN çok daha savunulabilir bir başlangıçtır.

---

## 9. Temel doğrulama kaynakları

- Gasse vd. — Exact Combinatorial Optimization with Graph Convolutional Neural Networks  
  https://arxiv.org/abs/1906.01629
- Tong vd. — Directed Graph Convolutional Network  
  https://arxiv.org/abs/2004.13970
- Schlichtkrull vd. — Modeling Relational Data with Graph Convolutional Networks  
  https://arxiv.org/abs/1703.06103
- Cai vd. — Line Graph Neural Networks for Link Prediction  
  https://doi.org/10.1109/TPAMI.2021.3080635

Bu kaynakların bazıları doğrudan OR çalışması değildir. Buradaki amaç model ailesinin gerçekten var olduğunu ve hangi graph yapısını işlediğini doğrulamaktır; OR kullanım gerekçesi problem temsilinden türetilir.
