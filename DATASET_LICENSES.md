# Dataset Licensing Notes

This file summarizes inferred dataset licensing constraints for this repository.

Legal note: this is a technical compliance aid, not legal advice.

## Scope

The project uses FER-style CSVs and merged training sets derived from multiple sources.
Code in this repository is MIT-licensed, but dataset rights are independent from code rights.

## Dataset Matrix (Inferred)

| Dataset | Source | Observed/Provided Terms | Commercial Use | Redistribution of Raw Data | Key Obligations | Confidence |
|---|---|---|---|---|---|---|
| CAER-S | https://opendatacommons.org/licenses/odbl/1-0/ | ODbL 1.0 (database license) | Yes (ODbL allows) | Conditional | Attribution, keep ODbL notice, share-alike on public derivative databases; provide derivative DB or change set/method when publicly used | Medium |
| Emotic (Kaggle mirror) | https://www.kaggle.com/datasets/magdawjcicka/emotic | Page text says non-commercial research/education only; Kaggle card shows License: Unknown | Treat as No | Treat as No/Unclear | Research/education only, attribution/citation, do not republish images unless explicit permission exists | Medium |
| HECO | https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970141.pdf | Paper URL is not itself a license statement | Unknown | Unknown | Must confirm official dataset hosting terms before public redistribution | Low |
| CK+ | Not used in current requested setup | Not applicable to this release if unused | Not applicable | Not applicable | Keep out of compliance scope if excluded from training/eval artifacts | High |
| FER2013 (Kaggle msambare) re-labeled in this repo | https://www.kaggle.com/datasets/msambare/fer2013 | Kaggle lists "Database: Open Database, Contents: Database Contents" (ODC DBCL 1.0) | Likely Yes (under DBCL terms) | Conditional | Keep attribution/license notice; DBCL obligations apply to database contents; re-labeling does not remove original obligations | High |

## Practical Interpretation

1. Merged datasets inherit the strictest upstream constraints.
2. Mixing open-database licensed sources (for example ODbL/DBCL style obligations) with non-commercial or unknown-license sources can make public redistribution of merged row-level data legally risky or incompatible.
3. Publishing model code is usually simpler than publishing the underlying merged datasets.

## Release Policy Options

### Strict (Recommended)

1. Publish: code, configs, training scripts, metrics, and model cards.
2. Do not publish: raw third-party data, merged CSV rows, or derived row-level databases.
3. Add a notice: users must obtain each dataset from original providers under original terms.

### Medium

1. Publish checkpoints only if no source terms prohibit it.
2. Include dataset-attribution block and explicit non-commercial warning where applicable.
3. Keep merged training CSVs private unless all upstream licenses are confirmed compatible.

### Permissive (Only after verification)

1. Publish derivative databases only when every source permits it and obligations are met.
2. For ODbL-covered derivative databases, include ODbL notice and required access to derivative DB or modification method.
3. Document exact provenance and license compatibility mapping for each merged row source.

## Minimum Compliance Checklist

1. Record exact source URL and license text for each dataset version used.
2. Confirm whether commercial use is allowed per source.
3. Confirm whether redistribution is allowed for raw images, annotations, and derived CSVs.
4. Add required citations and attribution notices.
5. Add release README note: model/data rights differ; data must be re-downloaded by users.
6. Keep a frozen manifest of dataset versions and hashes used for each experiment.

## Suggested Repository Notice Snippet

"This repository code is MIT-licensed. Models are trained on third-party datasets
that are subject to their own licenses and terms. Users must obtain datasets from
original sources and comply with original dataset terms (including attribution,
non-commercial limits, and share-alike obligations where applicable)."