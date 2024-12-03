# Cat-vs.-Cat-Loaf
# Introduction
This project investigates the classification of images
into two distinct categories: ”Cat” and ”Cat Loaf,” using Con-
volutional Neural Networks (CNNs). The project is inspired by
the nuanced challenge of recognizing animal postures, specifically
differentiating traditional cat poses from loaf-like compact posi-
tions. The dataset contains 646 RGB images, evenly split between
the two classes. Images were preprocessed through resizing,
normalization, and data augmentation to enhance generalization.
The CNN architecture employed consists of convolutional, max-
pooling, and dense layers, achieving a validation accuracy of
approximately 66%. This project highlights the potential of CNNs
in solving posture-based classification tasks, emphasizing the
importance of feature extraction and augmentation in improving
model performance. The study also discusses challenges such
as subtle posture variations and dataset limitations, proposing
future improvements to address these issues.
# Convolutional Neural Networks (CNNs)
Convolutional Neural Networks (CNNs) are a class of
deep learning architectures specifically designed to process
and analyze image data. They excel at automatically learning
hierarchical features directly from raw pixel inputs, making
them a cornerstone of modern computer vision tasks. CNNs
are inspired by the biological processes of the visual cortex,
where neurons are sensitive to specific regions of an image,
enabling localized feature detection.
A CNN processes image data through a series of layers
that extract increasingly complex features. The architecture
typically consists of the following key components:
1) Convolutional Layers: Convolutional layers apply
learnable filters (kernels) to the input image, sliding
across spatial dimensions (height and width) to produce
feature maps. These feature maps encode spatial infor-
mation such as edges, textures, and shapes, which are
crucial for understanding image content. The convolu-
tion operation is mathematically expressed as:
O(i, j) =
M −1X
m=0
N −1X
n=0
I(i + m, j + n) · K(m, n)
where I represents the input image, K is the kernel,
and O is the output feature map. The kernel’s size and
stride determine how much the feature map is reduced,
while the activation function (e.g., ReLU) introduces
non-linearity, allowing the network to model complex
patterns.
2) Pooling Layers: Pooling layers downsample the spa-
tial dimensions of feature maps while retaining the
most significant information. This operation reduces
computational complexity, minimizes overfitting, and
improves feature invariance to translations. Common
pooling methods include:
• Max-Pooling: Retains the maximum value within
each pooling window, emphasizing strong activa-
tions.
• Average-Pooling: Computes the average value
within the pooling window, offering a smoother
representation.
Max-pooling is often preferred for its ability to highlight
prominent features while discarding background noise.
3) Fully Connected Layers: After the convolutional and
pooling layers extract spatial features, fully connected
layers (dense layers) process this information to make
predictions. These layers interpret the high-level features
by learning complex decision boundaries. The final fully
connected layer typically outputs class probabilities us-
ing a softmax (for multi-class classification) or sigmoid
activation function (for binary classification).

In this study, CNNs are used to distinguish between subtle
posture differences in ”Cat” and ”Cat Loaf” images. The
convolutional layers extract posture-specific features, such as
compactness or body outline, while pooling layers ensure these
features are invariant to minor variations in pose or alignment.
The fully connected layers interpret these features, producing
a probability score for binary classification. By leveraging
the hierarchical learning capabilities of CNNs, this project
effectively addresses the challenges of posture-based image
classification.

# DATASET OVERVIEW
## Dataset Description
The ”Cat vs. Cat Loaf” dataset consists of 646 RGB images,
evenly divided into two classes:
• Cat: 323 images of cats in traditional poses.
• Cat Loaf: 323 images of cats in a loaf posture, charac-
terized by a compact body with paws tucked underneath.
Images were resized to 100 × 100 pixels to standardize
input dimensions for the CNN. This uniformity simplifies the
learning process and ensures that the model focuses on feature
extraction rather than data variability.
## Relevance of the Dataset
This dataset presents a unique challenge in computer vision,
as the classification task relies on detecting subtle differences
in posture rather than entirely different objects. The problem
highlights the importance of spatial feature extraction in neural
networks and serves as a testbed for evaluating the robustness
of CNN architectures in nuanced classification tasks.
# METHODOLOGY
A. Data Preprocessing
To prepare the dataset for training, the following prepro-
cessing steps were applied:
• Resizing and Normalization: All images were resized
to 100 × 100 pixels, and pixel values were scaled to
the range [0, 1] to ensure uniform input and faster model
convergence.
• Data Augmentation: Random transformations, including
rotation (up to 30◦), width and height shifts (up to 20%),
and horizontal flips, were applied. These augmentations
increased dataset diversity, reduced overfitting, and im-
proved generalization.
B. Model Architecture
The CNN architecture consists of:
• Three convolutional layers with filter sizes of 32, 64, and
128, each followed by ReLU activation and max-pooling.
• A fully connected layer with 128 neurons and ReLU
activation.
• A dropout layer with a rate of 0.5 to prevent overfitting.
• A final dense layer with sigmoid activation for binary
classification.
C. Training and Validation
The dataset was split in an 80:20 ratio, with the majority
reserved for training and a smaller subset for validation. The
model was trained for 10 epochs using the Adam optimizer and
binary cross-entropy loss. Batch size was set to 32, balancing
training efficiency and memory requirements.
V. EXPERIMENTATION AND EVALUATION
A. Evaluation Metrics
The model’s performance was assessed using accuracy,
precision, recall, and F1-score. These metrics provided a com-
prehensive understanding of the model’s efficacy. A confusion
matrix was used to visualize the distribution of predictions
across the ”Cat” and ”Cat Loaf” classes, highlighting strengths
and potential areas for improvement.
B. Experimentation Steps
The experimentation process involved:
1) Loading and preprocessing the dataset, including nor-
malization and augmentation.
2) Training the CNN on the augmented dataset with an
80:20 split for training and validation.
3) Hyperparameter tuning to optimize learning rates, batch
sizes, and network depth.
4) Monitoring performance across epochs to identify po-
tential overfitting or underfitting.
5) Testing on unseen data to evaluate real-world applica-
bility.
# RESULTS AND DISCUSSION
After completing the training process, the model achieved
a validation accuracy of 66.15% and a validation loss of 0.62.
Figures 1 and 2 show the training and validation accuracy and
loss curves.
Fig. 1. Training and Validation Accuracy. The accuracy plot illustrates the
model’s learning progress over the epochs, showing improvement in both
training and validation accuracy.
Fig. 2. Training and Validation Loss
The results demonstrate the model’s ability to distinguish
between the two classes, though subtle posture differences and
a limited dataset size present challenges. Future improvements,
such as increasing the dataset size or using transfer learning,
could enhance performance.
