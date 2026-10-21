# Figure source data

| Figure | Scientific content | Principal tables |
|---|---|---|
| Figure1 | Design, sample/gene counts and validation partitions | `design_values.tsv`, nominal input gene lists |
| Figure2 | Sample/gene-wise reconstruction and strict OOF retrieval | Panels A–D and numerical distributions |
| Figure3 | Condition labels and sparse-input condition retrieval | Paired metric panels and neighbor fractions |
| Figure4 | Replicate residuals and LOCO within-condition residuals | Panels A–D and residual summaries |
| Figure5 | LOCO centroid shifts and strict OOF retrieval | Panels A–D and condition-level numerical sources |
| FigureS1 | Measured target variability quartiles | Annotated plot values and gene-variability metrics |
| FigureS2 | Computational seed robustness | Annotated plot values and seed-robustness metrics |
| FigureS3 | TopN sensitivity on common and bridge targets | Panels A–D and input-depth metrics |

`registry_links.tsv` is the authoritative connection between each plot cell and a
row of `reproducibility/result_registry.tsv`. Paths resolve from the repository
root. `plot_data_row` uses a zero-based data-row index (header excluded).
`source_selector` is a JSON dictionary selecting rows from `source_file`;
`source_column` and `operation` specify the reduction. Scientific counts and metric
values are preserved. Full numeric distributions are included where needed to
recreate plotted boxes, points and whiskers.

`numerical_sources/` contains inputs to the figure scripts; panel tables contain
the final displayed numerical cells. An annotated plot table is kept in preference
to an otherwise identical unannotated intermediate. Metadata `cell_type` labels are
biological conditions; numerical `true` column names mean measured expression.
