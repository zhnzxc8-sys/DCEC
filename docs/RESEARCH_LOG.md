# CVIP 2026 Research Log

## 2026-09-22

### Research Topic
Robust Image-Text Retrieval under Visual Degradations

### Current Stage
Environment setup and baseline preparation.

### Completed
- Python 3.11.9 installed.
- Git installed.
- Python virtual environment created.
- PyTorch installed successfully.
- PyTorch CPU execution verified.
- CUDA availability: False.

### Hardware
- Operating System: Windows
- Memory: approximately 16 GB
- GPU: Intel Iris Xe Graphics
- CUDA GPU: None

### Next Step
Establish the codebase and verify the complete research environment.
## 2026-09-23

### CLIP Smoke Test

Pretrained model:
openai/clip-vit-base-patch32

Device:
CPU

Input image:
data/samples/test.jpg

Candidate texts:
- a photo of a cat
- a photo of a car
- a photo of a building

Observed output:
- a photo of a cat: 0.6006%
- a photo of a car: 1.4547%
- a photo of a building: 97.9448%

### Interpretation
The pretrained CLIP inference pipeline was successfully executed on the local CPU environment.

The reported percentages are softmax probabilities among the three candidate texts and are not treated as benchmark retrieval metrics.

### Next Step
Implement and validate image-text retrieval evaluation metrics.
## 2026-09-23

### Flickr30K Clean Baseline Pipeline Smoke Test

Dataset:
Flickr30K Karpathy test split

Model:
openai/clip-vit-base-patch32

Local test scale:
10 images and 50 captions

Feature dimensions:
- Vision pooled representation: 768
- Image CLIP embedding: 512
- Text pooled representation: 512
- Text CLIP embedding: 512

Observed smoke-test metrics:
- I2T R@1: 100.0
- I2T R@5: 100.0
- I2T R@10: 100.0
- T2I R@1: 96.0
- T2I R@5: 100.0
- T2I R@10: 100.0

### Important Note
These results are generated from only 10 images and 50 captions and are used solely to validate the experimental pipeline. They are not formal benchmark results and must not be used as evidence in the final paper.

### Status
The Flickr30K -> CLIP -> embedding -> similarity -> Recall@K pipeline has been successfully validated locally.

### Next Step
Run the full Flickr30K Karpathy test split on a GPU server to obtain the formal clean CLIP baseline.
## Clean Baseline：CLIP ViT-B/32 on Flickr30K Karpathy Test

### 实验目的
建立后续视觉退化鲁棒性实验的统一无退化基线。该实验不进行训练或参数微调，仅使用预训练 CLIP ViT-B/32 提取图像与文本特征，并基于余弦相似度完成双向跨模态检索。

### 数据集与划分
- Dataset: Flickr30K
- Dataset source: noonamkha/flickr30k-karpathy
- Split protocol: Karpathy
- Evaluation split: test
- Number of images: 1000
- Captions per image: 5
- Number of captions: 5000

### 模型
- Model: openai/clip-vit-base-patch32
- Training / Fine-tuning: None
- Model weight format: safetensors

### 评价指标
采用双向 Recall@K：
- Image-to-Text: R@1, R@5, R@10
- Text-to-Image: R@1, R@5, R@10

### 实验结果
| Retrieval Direction | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Image-to-Text | 79.40 | 95.00 | 98.10 |
| Text-to-Image | 58.70 | 83.42 | 90.12 |

### 实验环境
- GPU: NVIDIA GeForce RTX 4090 D, 24 GB
- Python: 3.12.3
- PyTorch: 2.5.1+cu124
- CUDA Runtime: 12.4
- Transformers: 5.17.0
- Datasets: 5.0.1
- PyArrow: 25.0.1
- Code commit: f4634f5

### 结果说明
本实验结果作为后续视觉退化条件下跨模态检索性能变化的 Clean Baseline。该结果来源于完整 Flickr30K Karpathy test split，不使用训练、微调或测试集参数优化。正式结果文件、运行日志、环境信息以及代码和数据文件 SHA256 校验信息均已独立保存。

### 状态
Clean Baseline completed.
## Visual Degradation Sensitivity Evaluation

### 实验目的
在固定 CLIP ViT-B/32、Flickr30K Karpathy test 划分和既定 Recall@K 评价协议的条件下，系统评估常见视觉退化对图文跨模态检索性能的影响，为后续退化感知鲁棒方法设计提供实验依据。

### 固定实验设置
- Dataset: Flickr30K
- Dataset split: Karpathy test
- Number of images: 1000
- Captions per image: 5
- Number of captions: 5000
- Model: openai/clip-vit-base-patch32
- Training / fine-tuning: None
- Device: NVIDIA GeForce RTX 4090 D
- Code commit: 26f090c
- Default random seed: 2026

### Clean Baseline
| Retrieval Direction | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Image-to-Text | 79.40 | 95.00 | 98.10 |
| Text-to-Image | 58.70 | 83.42 | 90.12 |

### JPEG Compression
| Severity | Quality | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 75 | 78.50 | 95.40 | 98.40 | 58.08 | 83.16 | 89.42 |
| S2 | 50 | 78.70 | 94.70 | 97.90 | 57.44 | 82.30 | 88.28 |
| S3 | 25 | 77.30 | 94.30 | 97.80 | 58.72 | 83.72 | 90.02 |

### Gaussian Blur
| Severity | Radius | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 1.0 | 77.50 | 94.00 | 96.80 | 57.88 | 82.36 | 88.84 |
| S2 | 2.0 | 72.30 | 92.10 | 95.90 | 53.40 | 79.60 | 87.02 |
| S3 | 4.0 | 55.30 | 80.70 | 88.80 | 41.22 | 69.50 | 79.12 |

### Gaussian Noise

Gaussian Noise 为随机退化，因此使用 seed=2026、2027、2028 三次独立实验，并报告均值 ± 样本标准差。

| Severity | Sigma | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 0.02 | 78.40±0.44 | 94.57±0.23 | 98.10±0.17 | 58.56±0.16 | 83.79±0.19 | 89.94±0.05 |
| S2 | 0.05 | 78.07±0.15 | 94.23±0.46 | 97.80±0.10 | 58.33±0.17 | 83.53±0.24 | 89.75±0.21 |
| S3 | 0.10 | 77.13±0.76 | 93.63±0.12 | 97.17±0.06 | 57.01±0.22 | 82.33±0.22 | 89.01±0.33 |

### Low-Light
| Severity | Brightness Factor | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 0.75 | 79.50 | 94.70 | 98.30 | 57.80 | 83.06 | 89.40 |
| S2 | 0.50 | 78.50 | 94.20 | 97.70 | 57.06 | 82.30 | 88.82 |
| S3 | 0.25 | 75.40 | 92.50 | 96.60 | 56.42 | 80.88 | 87.72 |

### 主要实验现象
1. 不同视觉退化对 CLIP 跨模态检索性能的影响存在明显差异。
2. Gaussian Blur 的性能退化最显著，且随模糊强度增加呈明显下降趋势。
3. Gaussian Blur S3 条件下，I2T R@1 由 79.40 降至 55.30，下降 24.10 个百分点；T2I R@1 由 58.70 降至 41.22，下降 17.48 个百分点。
4. Low-Light 和 Gaussian Noise 对性能产生中等或较轻影响。
5. JPEG Compression 在当前设置下整体影响相对较小，并存在轻微非单调波动。该现象按原始实验结果保留，不进行人为修正。
6. Gaussian Noise 三随机种子实验的标准差整体较小，表明当前实验条件下结果具有较好的重复性。

### 阶段结论
当前实验仅用于识别基础 CLIP 在不同视觉退化条件下的敏感性差异，不据此宣称任何鲁棒方法已经有效。实验结果表明，高斯模糊造成的结构和高频视觉信息损失可能是当前跨模态检索系统的主要脆弱因素之一，后续方法设计将围绕退化感知和视觉语义嵌入稳健性进一步展开。

### 状态
Visual degradation sensitivity evaluation completed.
## Train / Validation / Test Isolation Audit

### Dataset Split
- Train: 29000 images
- Validation: 1014 images
- Test: 1000 images
- Captions per image: 5

### Integrity Check
- Train duplicate image IDs: 0
- Validation duplicate image IDs: 0
- Test duplicate image IDs: 0
- Invalid caption-count rows: 0
- Empty captions: 0

### Split Overlap
- Train ∩ Validation: 0
- Train ∩ Test: 0
- Validation ∩ Test: 0

### Data Integrity Records
- Train parquet shards: 8
- Validation parquet shards: 1
- Train and validation parquet SHA256 records have been generated.
- A programmatically generated split audit has been retained with the experiment provenance.

### Development Protocol
From the DCEC method-development stage onward, the Flickr30K Karpathy test split is frozen. Model training is restricted to the training split, while hyperparameter selection and checkpoint selection are restricted to the validation split. The test split is not used for training, hyperparameter tuning, checkpoint selection, or method modification.

### Status
Train / validation / test isolation verified.
## DCEC v1 Validation Evaluation

### 模型选择协议
DCEC v1 仅使用 Flickr30K Karpathy training split 进行训练，并仅使用 validation split 进行 checkpoint 选择。Test split 在方法开发阶段保持冻结。

Best checkpoint:
- Epoch: 3
- Validation primary score: 70.0828
- Seed: 2026

Validation primary score 定义为五个预先固定条件下：
(I2T R@1 + T2I R@1) / 2
的算术平均值。

固定 checkpoint 后，在完整 validation degradation sweep 上进一步评价模型。

### Full Validation Sweep

| Condition | Frozen CLIP Primary | DCEC v1 Primary | Delta |
| --- | ---: | ---: | ---: |
| Clean S0 | 69.2406 | 69.9507 | +0.7101 |
| JPEG S1 | 69.9112 | 71.1045 | +1.1933 |
| JPEG S2 | 69.9112 | 71.0059 | +1.0947 |
| JPEG S3 | 70.1381 | 71.0651 | +0.9270 |
| Gaussian Blur S1 | 69.4083 | 70.1282 | +0.7199 |
| Gaussian Blur S2 | 66.1736 | 68.3235 | +2.1499 |
| Gaussian Blur S3 | 49.6844 | 56.1341 | +6.4497 |
| Gaussian Noise S1 | 70.4043 | 70.9961 | +0.5917 |
| Gaussian Noise S2 | 70.0493 | 70.6509 | +0.6016 |
| Gaussian Noise S3 | 68.3728 | 69.3787 | +1.0059 |
| Low-Light S1 | 69.6351 | 70.0000 | +0.3649 |
| Low-Light S2 | 69.2406 | 70.4832 | +1.2426 |
| Low-Light S3 | 67.8797 | 69.4379 | +1.5582 |

### Aggregate Result
Across the 12 degraded validation conditions:

- Frozen CLIP mean primary score: 67.5674
- DCEC v1 mean primary score: 69.0590
- Absolute improvement: +1.4916
- Positive degradation conditions: 12 / 12
- Negative degradation conditions: 0 / 12

The largest validation improvement was observed under Gaussian Blur severity 3:
49.6844 → 56.1341, corresponding to +6.4497 primary-score points.

Clean validation performance was also retained and increased:
69.2406 → 69.9507, corresponding to +0.7101 points.

### Interpretation
Under the predefined validation protocol, the fixed DCEC v1 checkpoint produced positive primary-score changes across all evaluated degradation conditions. The largest gain occurred under severe Gaussian blur, consistent with the preceding degradation-sensitivity analysis in which blur produced the largest degradation of the Frozen CLIP baseline.

These results constitute validation-stage evidence only. They are not treated as final test-set conclusions.

### Development Decision
The DCEC v1 main architecture is frozen after validation. Subsequent development will focus on controlled ablation experiments rather than modifying the main architecture according to validation results.

The Flickr30K Karpathy test split remains frozen.

### Status
DCEC v1 validation completed.
Proceed to ablation study.