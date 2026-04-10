# Model Licensing Notes

This file summarizes licensing and redistribution constraints for model code,
third-party model dependencies, and trained checkpoints in this repository.

Legal note: this is a technical compliance aid, not legal advice.

## Scope

The project includes:
1. Model source code under this repository.
2. Third-party frameworks and model libraries used at runtime.
3. Optional pretrained backbones/weights fetched from third-party ecosystems.
4. Trained checkpoints produced from third-party datasets.

## Model Matrix (Inferred)

| Component | Local Path | Origin | Local Code License Status | External Terms to Check | Redistribution Guidance | Confidence |
|---|---|---|---|---|---|---|
| ResNet (custom implementation) | models/resnet.py | In-repo implementation based on ResNet architecture paper | Covered by repository MIT | None beyond repo license for this file | May redistribute source under MIT terms | High |
| ResNet regression variants | models/resnet_reg2.py | In-repo code + torchvision backbone API | Covered by repository MIT for local code | PyTorch/torchvision package licenses; pretrained weight terms when downloaded | Source redistribution usually fine; pretrained-weight distribution depends on upstream terms | Medium |
| ResNet50 pretrained backbone usage | models/resnet_reg2.py | torchvision.models.resnet50(pretrained=True) | Local wrapper code is MIT | Torchvision model/weights terms and notices | If distributing a checkpoint that includes pretrained initialization effects, retain upstream notices and verify allowed use | Medium |
| EfficientNet VAD wrapper | models/efficientnet_b0.py | In-repo wrapper using efficientnet_pytorch | Covered by repository MIT for wrapper code | efficientnet_pytorch package license and any linked pretrained weight terms if used later | Source redistribution fine under MIT for wrapper; verify third-party dependency and weights notices | Medium |
| MobileFaceNet VAD | models/mobilefacenet.py | In-repo implementation/adaptation | Covered by repository MIT | Any external code provenance if copied from third-party sources outside repo | Source redistribution fine under MIT for current in-repo code | Medium |
| Trained checkpoints | FER2013_ResNet/*.pth | Produced in this project from third-party datasets | Not automatically governed only by code MIT | Dataset terms from all training sources (see DATASET_LICENSES.md) | Treat as constrained by strictest dataset terms; avoid public release if source terms are non-commercial/unknown | High |

## Practical Interpretation

1. Code license and model-data rights are separate concerns.
2. Repository model code is MIT, but checkpoints inherit dataset-use constraints.
3. Third-party pretrained weights and libraries may require extra attribution/notices.

## Release Policy Options

### Strict (Recommended)

1. Publish model source code and training/inference scripts.
2. Publish checkpoint metadata (architecture, hyperparameters, metrics) without weights if terms are unclear.
3. Release weights only when all dataset and upstream weight terms permit redistribution.

### Medium

1. Publish selected checkpoints with explicit "research-only" notice when required by training data terms.
2. Include dependency and pretrained-backbone attribution block.
3. Add a provenance statement mapping each checkpoint to datasets used.

### Permissive (Only after verification)

1. Publish broad checkpoint sets when all data and upstream weight licenses are confirmed compatible.
2. Provide complete attribution and required notices for datasets, libraries, and pretrained sources.
3. Include an auditable training manifest for each released checkpoint.

## Minimum Compliance Checklist for Model Release

1. Confirm dataset permissions for checkpoint redistribution.
2. Confirm pretrained backbone/weight permissions where used.
3. Include LICENSE plus attribution section in release artifacts.
4. Add a model card listing training data sources and restrictions.
5. Add "no rights to underlying third-party datasets are granted" statement.

## Suggested Model Release Notice Snippet

"Model code in this repository is MIT-licensed. Released checkpoints may include
effects of training on third-party datasets and optional pretrained backbones,
which remain subject to their original licenses and terms. No rights to
third-party datasets are granted by this repository."
