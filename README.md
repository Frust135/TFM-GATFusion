# GATFuse

Detección de fusiones génicas a partir de RNA-seq mediante una red de atención sobre grafos (GATv2). Incluye también un baseline XGBoost sobre las mismas features de arista. Este repositorio contiene el código para reproducir el pipeline (grafo → entrenamiento → predicción).

## Requisitos

- Python 3.10+
- Dependencias: `pip install -r requirements.txt`
- Herramientas externas en el `PATH`:
  - `STAR` — necesario para `scripts/download_references.py` (build del índice) y para generar los ficheros de entrada de cada muestra (alineamiento).
  - `samtools` — opcional, solo se usa para indexar el FASTA de referencia si está disponible.

## Estructura del proyecto

```
graph_builder.py       Construye el grafo (nodos = genes, aristas = uniones quiméricas)
model.py                Arquitectura GATv2 (FusionPredictor), entrenamiento, checkpoints
data_parser/             Parsers de BAM, Chimeric.out.junction, ReadsPerGene.out.tab, GTF, Mitelman DB
utils/
  create_train_data.py    Etiquetado de aristas positivas a partir de fusiones conocidas
  postprocess.py           Filtros aplicados a las predicciones
scripts/
  download_references.py  Descarga genoma + anotación y construye el índice STAR
  train_single_graph.py   Entrena sobre una única muestra
  train_multi_graph.py    Entrena sobre varias muestras (TSV), con validación o CV
  train_xgboost.py        Entrena el baseline XGBoost
  predict.py               Predicción con un checkpoint GATv2
  predict_xgboost.py       Predicción con un modelo XGBoost
  scrape_mitelman.py       Descarga la tabla de pares de genes recurrentes de Mitelman
checkpoints/             Modelos ya entrenados (GATv2 y XGBoost) de ejemplo
benchmarks/               Predicciones de GATFuse y de otras herramientas sobre muestras
data/                     GTF de referencia y tabla Mitelman usados por defecto
cache/graphs/             Grafos ya construidos, cacheados en disco (se genera solo)
```

## Entradas necesarias por muestra

Cada muestra RNA-seq alineada con STAR aporta:

- BAM ordenado (`Aligned.sortedByCoord.out.bam`)
- `Chimeric.out.junction`
- `ReadsPerGene.out.tab`
- Un GTF de anotación (el mismo usado para alinear)

## 1. Descargar genoma de referencia + anotación

```bash
python scripts/download_references.py ASSEMBLY+ANNOTATION
python scripts/download_references.py GRCh38+GENCODE38
```

## 2. Entrenar

### 2.1 Una única muestra

```bash
python scripts/train_single_graph.py \
    --bam_file sample.bam \
    --chimeric_file sample.Chimeric.out.junction \
    --reads_per_gene_file sample.ReadsPerGene.out.tab \
    --gtf_file ref_annot.gtf \
    --positive_fusion TMPRSS2:ERG \
    --output_model checkpoints/sample.pt
```

`--positive_fusion` puede repetirse para varias fusiones, o usar `--positive_fusions_file` con un archivo de una fusión `DONOR:ACCEPTOR` por línea.

### 2.2 Varias muestras vía archivo TSV (recomendado)

Archivo TSV/CSV con una fila por muestra (ver `training_plan.tsv` como ejemplo real). Columnas:

| Columna | Obligatoria | Descripción |
|---|---|---|
| `bam_file` | sí | ruta al BAM ordenado |
| `chimeric_file` | sí | ruta a `Chimeric.out.junction` |
| `reads_per_gene_file` | sí | ruta a `ReadsPerGene.out.tab` |
| `positive_fusions` | una de las dos | fusiones separadas por `;`, formato `DONOR:ACCEPTOR` (opcionalmente `@chrA:posA-chrB:posB`) |
| `gtf_file` | no | GTF específico de la fila; si se omite se usa `--gtf_file` |

```bash
python scripts/train_multi_graph.py \
    --manifest training_plan.tsv \
    --gtf_file data/ref_annot.gtf \
    --output_model checkpoints/final_model.pt
```

Opciones más usadas (`--help` para la lista completa):

- `--cv_folds N` — validación cruzada estratificada en N folds en vez de un único split (`--val_split`).
- `--mitelman_file data/mitelman_fusions.tsv` — añade features de recurrencia Mitelman (debe usarse igual en predicción).
- `--num_workers N` — construye los grafos de las N muestras en paralelo.
- `--graph_cache_dir` / `--no_cache` — cachea los grafos ya construidos entre ejecuciones.
- `--epochs`, `--lr`, `--hidden_dim`, `--heads`, `--num_gnn_layers`, `--dropout`, `--batch_size`, `--patience` — hiperparámetros del modelo/entrenamiento.
- `--resume_from checkpoint.pt` — continúa el entrenamiento desde un checkpoint existente.

### 2.3 Baseline XGBoost

```bash
python scripts/train_xgboost.py \
    --manifest training_plan.tsv \
    --gtf_file data/ref_annot.gtf \
    --output_model checkpoints/xgboost_model.json
```

Genera `checkpoints/xgboost_model.json` (modelo) y `checkpoints/xgboost_model.meta.json` (métricas, umbral e hiperparámetros).

## 3. Predecir fusiones en una muestra nueva

### GATv2

```bash
python scripts/predict.py \
    --bam_file sample.bam \
    --chimeric_file sample.Chimeric.out.junction \
    --reads_per_gene_file sample.ReadsPerGene.out.tab \
    --gtf_file ref_annot.gtf \
    --model checkpoints/final_model.pt \
    --output predictions.tsv
```

### XGBoost

```bash
python scripts/predict_xgboost.py \
    --bam_file sample.bam \
    --chimeric_file sample.Chimeric.out.junction \
    --reads_per_gene_file sample.ReadsPerGene.out.tab \
    --gtf_file ref_annot.gtf \
    --model checkpoints/xgboost_model.json \
    --output predictions.tsv
```

Ambos aceptan `--threshold` (si se omite, usa el threshold guardado en el checkpoint) o `--top_k` para quedarte con las N aristas de mayor score. Si se uso `--mitelman_file` en el entrenamiento, también se debe usar al predecir.

Salida (TSV): `donor_gene`, `acceptor_gene`, `score`, `split_reads`, `chr_donor`, `brkpt_donor`, `strand_donor`, `donor_region`, `chr_acceptor`, `brkpt_acceptor`, `strand_acceptor`, `acceptor_region`, `pct_canonical` (solo GATv2), `fusion_type`.

`predict.py` aplica por defecto un post-procesado que descarta fusiones intragénicas, read-through y uniones inter-cromosómicas no canónicas. Se controla con `--no_postprocess`, `--postprocess_annotate_only`, `--keep_intragenic`, `--keep_readthrough`, `--keep_blacklisted`, `--keep_noncanonical`.

## 4. (Opcional) Actualizar la base de datos Mitelman

```bash
python scripts/scrape_mitelman.py --output data/mitelman_fusions.tsv
```

Descarga y agrega los pares de genes recurrentes de la Mitelman Database en un TSV (`gene_a`, `gene_b`, `n_cases`) para usar con `--mitelman_file`.

## Benchmarks

`benchmarks/` contiene, por muestra (identificadas por su accession SRR), las predicciones ya generadas de GATFuse (`gatv2/`, `xgboost/`) junto a las de otras herramientas de detección de fusiones usadas como referencia en el TFM (`ARRIBA/`, `STAR-Fusion/`, `fusioncatcher_out/`). Son resultados, no código a ejecutar.

## Checkpoints incluidos

- `checkpoints/final_model.pt` — modelo GATv2 entrenado.
- `checkpoints/xgboost_model.json` + `checkpoints/xgboost_model.meta.json` — modelo XGBoost entrenado y sus metadatos.

## Caché de grafos

Construir un grafo desde BAM/Chimeric/ReadsPerGene es el paso más costoso del pipeline. Todos los scripts de entrenamiento y predicción cachean el grafo ya construido en `cache/graphs/` (clave = hash de las rutas/tamaños de los ficheros de entrada + parámetros de construcción). Usar `--no_cache` o `--graph_cache_dir ""` para desactivarlo.
