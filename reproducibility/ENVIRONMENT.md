# Top2All v1.0.0 computational environments

Figure/source-table validation was exercised with Python 3.12, Matplotlib 3.11.2,
NumPy 2.5.4, pandas 2.2.3, pypdf 6.10.0, and Pillow 13.0.0. The figure dependencies
were retained with the final local figure-generation environment. `requirements.txt`
records this environment. The release-integrity validator itself uses only the
Python standard library.

Arial must be installed separately and discoverable by Matplotlib for the original
figure typography. It is not redistributed. Output timestamps, PDF metadata and
SVG element identifiers can differ between runs; compare numerical plot tables and
rendered content, rather than expecting byte-identical newly generated exports.
The committed exports were regenerated from the selected final sources with
manuscript-facing label/title corrections; scientific values and nontext geometry
are preserved.

`requirements-models.txt` records the core versions from a locally retained model
environment export (PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.0, pandas 2.3.3).
Use a separate environment for model training. The retained export listed CUDA
12.4 packages, but there is no complete lockfile cryptographically bound to every
final training run. Those versions are a reference, not a claim of an exact
historical environment for all runs. Model execution was not repeated during
release finalization. Exact curated inputs and formal predictions are included in
the core archive. Project-trained checkpoints are excluded from that archive by
scope, and pretrained encoder/dictionary resources remain external dependencies.

The reconstruction/scale seed pairs are 42/1234, 43/1235 and 44/1236. Original
NumPy/PyTorch seeding and encoder training/evaluation modes are retained. Frozen
encoder dropout is active during reconstruction-head training. Hardware and library
changes may affect retrained numerical outputs even with the same seeds.

Figure terminology was aligned to the final manuscript on 2026-10-21. All eight
figures were regenerated with the retained figure environment. Nontext SVG drawing
primitives, marker coordinates, paths and panel geometry match the pre-edit exports
exactly, including Figure1. Only visible labels, manuscript-facing titles and
normal export metadata changed. The previous Figure1 pixel discrepancy came from
the older export's Matplotlib 3.10.8 text rendering versus the retained 3.11.2
renderer; its nontext geometry is unchanged. Current exports now have a matching
regeneration recipe. Numerical schemas may retain internal true/truth identifiers;
those do not appear as inappropriate displayed measurement or condition labels.
