# A Locality-Aware Bruck Allgather (MPI)

Implementación en **C++/MPI** de:
- **Bruck Allgather** (baseline)
- **Locality-Aware Bruck Allgather** (optimizado para clusters jerárquicos: intra-nodo vs inter-nodo)

Inspirado en el enfoque *locality-aware* para minimizar comunicación **no-local** (inter-nodo) en `MPI_Allgather`. (Paper: Bienz, Gautam, Kharel, 2022)

---

## Setup experimental (Cluster)

Las pruebas se ejecutaron en un **cluster de 16 computadoras (16 nodos)** para evaluar desempeño y correctitud en un entorno distribuido real.

---

## Idea

El Bruck clásico usa pasos tipo \(\log_2(P)\) pero no distingue si el tráfico es intra-nodo o inter-nodo.  
La versión **locality-aware** organiza procesos por “regiones” (p.ej. nodo) para **reducir mensajes no-locales** intercambiándolos por más comunicación local cuando conviene.

---

## Archivos del repo

- `bruck.cpp` : Bruck Allgather (baseline)
- `localityAwareBruck.cpp` : Locality-Aware Bruck Allgather
- `local.cpp` y carpeta `local/` : utilidades / soporte local
- `prueba.cpp` y carpeta `prueba/` : pruebas / validación / benchmark
- `localityAwareBruckPrueba.cpp` : pruebas / benchmark del locality-aware

---

## Ejemplos

> Ejecución en el cluster (usando `hostfile` con los 16 nodos):

```bash
mpirun -np 16 --hostfile hosts.txt ./bruck_test
mpirun -np 16 --hostfile hosts.txt ./labruck_test
```

> Para ver el efecto de localidad, usa el mismo `P` variando procesos por nodo (`ppn`), por ejemplo:
- `P=16, ppn=4` (4 nodos × 4 procesos)
- `P=16, ppn=1` (16 nodos × 1 proceso)

---

## Validación

Se compara contra `MPI_Allgather` para verificar correctitud:

- `Bruck: OK`
- `Locality-Aware Bruck: OK`

---

## Resultados


**Figura 1.** Speedup del Locality-Aware Bruck vs baseline (MPI nativo).  
![Figura 1 - Speedup](imgs/speedup.png)

**Figura 2.** Comparación de tiempos (escala logarítmica).  
![Figura 2 - Tiempos](imgs/tiempos_log.png)

---

## Evidencia (runs)

**Figura 3.** Ejecución / validación (ejemplo 1).  
![Figura 3 - Run 1](imgs/run_1.png)

**Figura 4.** Ejecución / validación (ejemplo 2).  
![Figura 4 - Run 2](imgs/run_2.png)

---

## Créditos

Paper base: Bienz, Gautam, Kharel — *A Locality-Aware Bruck Allgather* (EuroMPI/USA 2022).  
Autores del repo: ver contributors.
