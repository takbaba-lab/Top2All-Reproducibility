# External resources and redistribution boundaries

The project's MIT license covers original code and accompanying release
documentation. It does not grant rights to external model resources or imply that
project-data redistribution has already received author approval. Proposed project
inputs, predictions, checkpoints and evaluations remain subject to author review.

`THIRD_PARTY_RESOURCES.tsv` records 17 external resources with provider URL,
version/identity, available SHA-256, hash-verification status and terms status.
No third-party resource is staged for the Zenodo deposit.

- **Mouse-Geneformer encoder:** the [official project](https://github.com/machine-perception-robotics-group/Mouse-Geneformer)
  links its [base-model archive](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view).
  Four file hashes from the analysis identify required local bytes. The original
  upstream revision and exact-byte redistribution terms were not retained/verified;
  a current provider download is not assumed identical to those hashes.
- **Two token dictionaries:** exact filenames and production hashes are recorded.
  The official project is the acquisition starting point; the exact original
  distribution/version and its redistribution terms remain unresolved. Obtain from
  the authors/provider and verify identity; do not bundle them by assumption.
- **Python packages:** package versions and provider URLs are recorded from the
  figure and reference model environments. No software distributions/wheels are
  copied into the archive; package-file hashes were not retained. Package licenses
  must come from their actual provider distributions.
- **Arial:** the locally installed font was hashed for rendering identity. No font
  bytes are staged or redistributed. See the [font provider's information](https://learn.microsoft.com/en-us/typography/font-list/arial)
  and the terms accompanying the user's licensed installation.
- **Raw RNA-seq:** [DDBJ DRA015952](https://ddbj.nig.ac.jp/resource/sra-submission/DRA015952)
  is referenced by accession. No raw reads are copied; consult the source record
  for the data's terms. Curated project expression inputs are a separate asset class.

The Mouse-Geneformer publication is [Ito et al., 2025](https://doi.org/10.1371/journal.pgen.1011420).
Its article terms are not treated as redistribution permission for separately
hosted model weights or dictionary files. No pretrained model was downloaded during
this local asset audit.
