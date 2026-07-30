# CVLAB Summer Project

## Stage 1: Classification Baselines

## Goal

The goal of Stage 1 is to establish a reliable and reproducible classification pipeline using frozen pretrained encoders. No flow-matching component is used at this stage.

Three classification baselines are considered:

1. linear probing;
2. classification using image-derived class prototypes;
3. zero-shot CLIP classification using text-derived class prototypes.

You will implement the linear probe and choose one of the two prototype-based branches. Thus, each group will work with two of the three baselines.

The selected prototype branch will be used in Stage 2, while the linear-probe setting will be used in Stage 3.

For all experiments, keep the encoders frozen, extract and cache the image features once, and perform classifier training and evaluation using the cached features.

## Datasets and Splits

Choose two out of the following three datasets:

| Dataset | Classes | Images |
|---|---:|---:|
| DTD | 47 | 5,640 |
| FGVC-Aircraft | 100 | approximately 10,000 |
| Oxford Flowers-102 | 102 | approximately 8,000 |

Use the following protocol:

- Use all classes and the official training, validation, and test splits.
- For DTD, use official partition 1.
- For FGVC-Aircraft, use the variant annotation level.
- Do not merge the training and validation splits.
- Use the full validation split for model selection, if needed.
- Use the full test split only for final evaluation.

For methods using labeled training examples, evaluate

$$
K \in \{5, 10, \text{full}\},
$$

where K is the number of selected training images per class. For K = 5 and K = 10, sample balanced subsets from the official training split using seeds {0, 1, 2}. The full setting uses the complete official training split.

## Frozen Encoders

Use publicly available pretrained checkpoints and the preprocessing associated with each checkpoint. All encoder parameters must remain frozen.

- **ImageNet-1K-pretrained ResNet-18:** use on both selected datasets. Use the standard pretrained model available through torchvision, and extract the 512-dimensional representation before the final classification layer.
- **DINOv2 ViT-S/14:** use on one of the two selected datasets, chosen by the group. Use a publicly available pretrained checkpoint and extract the final class-token representation.
- **CLIP RN50:** use only if the zero-shot CLIP branch is selected. Use the frozen CLIP image and text encoders.

Thus, the linear-probe experiments are performed with ResNet-18 on both datasets and are repeated with DINOv2 ViT-S/14 on one dataset of your choice. If the image-prototype branch is selected, repeat the prototype experiments using the same encoder settings.

Cache the training, validation, and test features before training the classifiers.

## Required Baseline: Linear Probe

Train a multiclass linear classifier

$$
s = Wz + b,
$$

where z is the frozen image feature. Only W and b are trained. Use softmax cross-entropy loss.

A suggested initial configuration is:

|  |  |
|---|---|
| Optimizer | AdamW |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-4}$ |
| Batch size | 64 |
| Maximum epochs | 200 |
| Checkpoint selection | Highest validation accuracy |

This is only a suggested baseline configuration. There is no need for an extensive hyperparameter search. The objective is to obtain a reasonable and stable linear probe, since the main later comparison will concern the FM layer. If the suggested setting behaves poorly, adjust it using the validation results and report the changes.

For the linear probe:

- Run each training-set size 3 times.
- For the 5-shot and 10-shot settings, use the corresponding subset seeds.
- For the full setting, use 3 classifier-initialization seeds.
- For one representative 10-shot run per dataset-encoder combination, show the training and validation loss curves.

## Choose One Prototype-Based Branch

### Option A: Image-Derived Class Prototypes

Use the frozen features produced by the same pretrained encoders described above: ImageNet-1K-pretrained ResNet-18 on both selected datasets, and DINOv2 ViT-S/14 on one dataset of your choice.

First $L_2$-normalize each feature. For every class $c$, compute

$$
\mu_c = \operatorname{normalize}\left(\frac{1}{|S_c|}\sum_{i \in S_c}\operatorname{normalize}(z_i)\right),
$$

where $S_c$ is the selected training subset for class $c$.

Classify using cosine similarity:

$$
\hat{y} = \arg\max_c \cos(z, \mu_c).
$$

Evaluate the 5-shot, 10-shot, and full settings.

- For the 5-shot and 10-shot settings, use the 3 subset seeds.
- The full-data result requires one run.

### Option B: Zero-Shot CLIP

Use the publicly available pretrained CLIP RN50 model, with both its image encoder and text encoder frozen.

Construct one text prototype per class using the relevant prompt:

- DTD: `a photo of a {class} texture`
- FGVC-Aircraft: `a photo of a {class} aircraft`
- Flowers-102: `a photo of a {class} flower`

Normalize the image and text embeddings and classify using

$$
\hat{y} = \arg\max_c \cos(z, t_c),
$$

where $t_c$ is the text embedding for class $c$.

Zero-shot CLIP does not use labeled training images and therefore produces one result per dataset.

## Evaluation and Results

Use top-1 accuracy on the complete official test split.

Report the results as follows:

- For 5-shot and 10-shot experiments, report the mean and standard deviation over 3 runs.
- For the full linear probe, report the mean and standard deviation over 3 initialization seeds.
- The full image-prototype result and the zero-shot CLIP result require one run.

You should be prepared to present and discuss:

1. **Accuracy table.** Report top-1 accuracy for every dataset, encoder, training-set size, and implemented baseline.
2. **Accuracy versus training-set size.** Plot accuracy for the 5-shot, 10-shot, and full settings, including error bars. If you selected zero-shot CLIP, its accuracy may be shown as a horizontal reference line.
3. **Training curves.** Show representative training and validation loss curves for the linear probe. The curves should demonstrate that training is stable and indicate whether substantial overfitting occurs.
4. **Confusion matrices.** Show one row-normalized confusion matrix for a representative setting on each selected dataset. Choose a setting that is useful for understanding the main classification errors.
5. **Feature visualizations.** Visualize the features of a readable subset of approximately 8–10 classes using PCA, t-SNE, or another suitable two-dimensional projection.

For the feature visualizations:

- Use the same selected classes, test examples, and class colors when comparing encoders or methods on the same dataset.
- For ResNet-18 and DINOv2, show test-image features together with the image-derived class prototypes.
- For CLIP, show test-image embeddings together with the corresponding text prototypes.
- When prototypes are displayed, fit the projection jointly to the image features and prototypes shown in the plot.
- Treat two-dimensional feature plots as qualitative visualizations rather than direct measures of classification performance.

You should be able to reproduce the results, explain the experimental protocol, and discuss the main observations from both the quantitative results and the visualizations.
