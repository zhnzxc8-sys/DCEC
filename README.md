# DCEC: Research Code and Experimental Evidence

This repository snapshot was prepared from `DCEC_FINAL_RESEARCH_BACKUP_20260929.tar.gz` for academic peer review. It contains the archived v11 project, the local TCR ViT-B/32 baseline working copy, results, logs, configurations, 38 adapter checkpoint files, environment records, and the checked English manuscript.

## Start here

- [Checked English manuscript](paper/DCEC_English_Advisor_Checked_20261009.pdf)
- [Framework](figures/DCEC_Framework_Advisor_20261009.svg)
- [Validation comparison table](results/paper_ready_20260928/TABLE_MAIN_21_CONDITIONS_20260928.csv)
- [Final-test summary](results/flickr30k_final_test/summary.json)
- [Training and selection records](results/multiseed/)
- [Efficiency records](results/journal_efficiency/)
- [Source code](src/) and [configurations](configs/)
- [Package validation](provenance/PACKAGE_VALIDATION.json)

## Scope and verification

The original archive contained 1,678 regular files. Their names and sizes matched the supplied inventory. Python syntax and JSON/JSONL parsing checks passed for 107 Python files, 278 JSON files, and 38 JSONL files; 18 CSV files were parsed. The 22 final-test condition files reproduced the archived summary: frozen mean 63.5109091, DCEC mean 65.0240909, population standard deviation across three seed means 0.0569701, and 22 condition-level wins. This verifies arithmetic against archived metrics, not independent reproduction from images. No training, inference, or checkpoint deserialization was performed.

The four modules absent from the earlier handoff (`dcec_dataset.py`, `dcec_model.py`, `eval_dcec_validation.py`, and `retrieval_metrics.py`) are present in this archive.

## Reproduction prerequisites and limitations

Raw dataset images, annotations, split manifests, and pretrained CLIP weights are not bundled here. Original server paths in configurations must be adapted. Dependency and hardware records are preserved in `DCEC_FINAL_BACKUP_META_20260929/`; the archived environment has not been recreated or tested. A complete end-to-end reproduction has not been verified.

Some configuration files retain historical `UNTRAINED` labels even though matching trained artifacts exist. Completion must be assessed from the relevant logs, metrics, checkpoints, and protocol records. Historical smoke runs and extension experiments are included and must not be treated as completed manuscript evidence merely because their files exist. Validation and final-test conditions are separate; final-test evidence must not be used for tuning.

Baseline repeat structures differ, so small mean differences do not establish statistical superiority. See the manuscript for other limitations. The manuscript's earlier note about metric rounding is preserved; archived unrounded final-test metrics are now available for review.

## Backup coverage

This package contains the final archive only. The legacy archive (881 regular files), residual-history archive (2,829 regular files, including nested older archives), and the two additional September 28 backups are not integrated. Their inventory records are provided where available. Comprehensive coverage of all historical experiments remains unverified. Original backup archives can be attached as GitHub Release assets separately after review.

## Packaging changes

Original code and experiment records were copied without editing. The v11 project directory was mapped to the repository root; baseline and environment directory names were retained. Git internals, Python caches, and three redundant handoff ZIPs were excluded; every exclusion and source mapping is recorded in `provenance/`. The upload `.gitignore` permits archived logs, results, and adapter checkpoints to be committed; the original ignore file is preserved. A basic credential-pattern scan of extracted text files found no matches; this is not an exhaustive privacy audit.

Original archive SHA-256: `0b1bd71408643b21e1f56ed8ed6e87dc32f54d2a3c7c42b61d444ab2bdabdc95`. This digest was computed from the received bytes and has not been compared with the separately stored backup checksum record.

See [third-party notices](THIRD_PARTY_NOTICES.md) for baseline attribution.
