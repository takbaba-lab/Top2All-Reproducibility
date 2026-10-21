# Computational environments

Figure/source-table validation was exercised with Python 3.12, Matplotlib 3.11.2,
NumPy 2.5.4, pandas 2.2.3, pypdf 6.10.0, and Pillow 13.0.0. The figure dependencies
were retained with the final local figure-generation environment. `requirements.txt`
records this environment. The release-integrity validator itself uses only the
Python standard library.

Arial must be installed separately and discoverable by Matplotlib for the original
figure typography. It is not redistributed. Output timestamps, PDF metadata and
SVG element identifiers can differ between runs; compare numerical plot tables and
rendered content, rather than expecting byte-identical newly generated exports.
The committed exports are byte-identical to the selected final manuscript exports.

`requirements-models.txt` records the core versions from a locally retained model
environment export (PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.0, pandas 2.3.3).
Use a separate environment for model training. The retained export listed CUDA
12.4 packages, but there is no complete lockfile cryptographically bound to every
final training run. Those versions are a reference, not a claim of an exact
historical environment for all runs. Model execution was not repeated during
release packaging because encoder assets and complete input profiles/checkpoints
were not available in this local release workspace.

The reconstruction/scale seed pairs are 42/1234, 43/1235 and 44/1236. Original
NumPy/PyTorch seeding and encoder training/evaluation modes are retained. Frozen
encoder dropout is active during reconstruction-head training. Hardware and library
changes may affect retrained numerical outputs even with the same seeds.

Visual regeneration check: Figures 2–5 and S1–S3 reproduce the committed PNG pixels
exactly in the retained figure environment. Figure1 recreates the same schematic
content and dimensions, with some icon/text placement differences from the frozen
export (0.89% of pixels differ); the supplied schematic source is therefore not a
byte/pixel-exact layout recipe. The committed final Figure1 remains the reference
for its exact design. This does not change sample partitions or numerical results.
The original figure labels `true`/`True` mean measured expression, and `state`
refers to transcriptomic/condition structure; these labels are retained in the
selected final exports. README and biological-condition descriptions use the
manuscript's condition terminology. No data or analysis alternatives are implied.
