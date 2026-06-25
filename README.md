# mapper-nd

> **Topological Data Analysis with the Mapper algorithm — implemented from scratch.**
> Most Mapper implementations project to the line (R) or the plane (R²) and return a
> **graph**. This one filters to an **arbitrary-dimensional space Rⁿ**, so the output is a
> full **simplicial complex** (edges, triangles, tetrahedra, …), not just a 1-skeleton.

![CI](https://img.shields.io/badge/CI-ruff%20%2B%20pytest-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Status:** Phases 1–2 of 5 complete — datasets and the **Mapper engine** (filters, cover,
nerve) are in place and tested on known shapes. The interactive visualiser and the diabetes
case study land in the phases below.

---

## What is Mapper?

Mapper turns a point cloud into a compact shape that captures its **topology**:

1. **Filter** — project the data with a function `f: X → Rⁿ` (eccentricity, graph
   Laplacian eigenvectors, a coordinate, …).
2. **Cover** — cover the image with **overlapping hypercubes**.
3. **Cluster** — cluster the points falling in each hypercube; every cluster becomes a
   **vertex**.
4. **Nerve** — connect vertices whose clusters share points. With an n-dimensional filter,
   up to 2ⁿ hypercubes overlap, so the nerve contains simplices up to dimension 2ⁿ−1.

The result is a simplicial complex that approximates the shape of the data — a circle
collapses to a loop, a sphere to a hollow shell, and (the headline demo) the
**Reaven–Miller diabetes study** to a core with two flares, one per diabetes subtype.

## Datasets

Self-contained and reproducible. Synthetic clouds are deterministic from a single seed;
the diabetes study is bundled as a CSV.

| Dataset | Points | Space | Known shape |
|---------|-------:|-------|-------------|
| `circle` | 400 | R² | one loop (S¹) |
| `torus` | 600 | R³ | two independent loops (S¹×S¹) |
| `sphere` | 400 | R³ | hollow shell (S²) |
| `diabetes` | 145 | R⁵ | central core + two flares |

```bash
uv sync --dev
uv run mapper-nd datasets    # list the bundled datasets and their shapes
```

The synthetic manifolds are the **ground truth** for validating the algorithm (Phase 4):
the recovered topology must match the known shape.

## The engine

The pipeline is configurable: pick a **filter**, a **cover** and a **clusterer**, then `fit`.

```python
from sklearn.cluster import DBSCAN
from mapper_nd.core import Mapper, HypercubeCover, GraphLaplacian
from mapper_nd.datasets import sphere

mapper = Mapper(
    filter=GraphLaplacian(kernel="rbf", n_components=2),  # project to R^2
    cover=HypercubeCover(n_intervals=5, overlap=0.5),     # overlapping squares
    clusterer=DBSCAN(eps=0.4, min_samples=3),
)
result = mapper.fit(sphere(300))
print(result.complex.simplex_counts())      # e.g. {0: 20, 1: 67, 2: 65, 3: 17}
print(result.complex.n_connected_components())
```

Because the lens is 2-dimensional, the nerve contains **triangles and tetrahedra**, not just
edges. Filters available: `Projection`, `Eccentricity`, `Statistic`, `GraphLaplacian`.

## Project layout

```
src/mapper_nd/
  config.py          # Pydantic settings (.env)
  log.py             # structlog setup
  datasets/          # reproducible point clouds + the diabetes study
  core/              # the Mapper engine: filters, cover, complex (nerve), mapper
  cli.py             # typer CLI
tests/               # pytest
data/diabetes.csv    # Reaven & Miller (1979), bundled
```

## Roadmap

- [x] **F1** — Repo scaffold + datasets (synthetic clouds + diabetes study)
- [x] **F2** — Core library: Mapper from scratch (filters, cover, nerve), arbitrary-dim Rⁿ
- [ ] **F3** — Interactive Dash/Plotly app: build the simplicial complex live in 2D/3D
- [ ] **F4** — Validation: recovered topology vs. known shape + the diabetes case study
- [ ] **F5** — README polish + CI + Docker, wired end to end

---

## Resumen (ES)

**Análisis de Datos Topológico con el algoritmo Mapper, implementado desde cero.** Mapper
reduce una nube de puntos a un **complejo simplicial** que captura su forma topológica:
una función filtro proyecta los datos, se recubre la imagen con **hipercubos solapados**,
se aplica *clustering* en la preimagen de cada hipercubo (cada cluster = un vértice) y se
conectan los vértices cuyos clusters comparten puntos (el *nervio*).

**Diferenciador:** la mayoría de implementaciones proyectan a R o R² y devuelven un
**grafo**; esta proyecta a **dimensión arbitraria Rⁿ**, generando un complejo simplicial
completo (aristas, triángulos, tetraedros…) de dimensión hasta 2ⁿ−1. La demo estrella
reproduce el estudio real de diabetes de **Reaven–Miller**: Mapper revela un núcleo central
con dos "brotes", uno por cada subtipo de diabetes.

Basado en el Trabajo Fin de Máster en Matemáticas del autor (Universidad de Málaga, 2022).

## Licencia

MIT
