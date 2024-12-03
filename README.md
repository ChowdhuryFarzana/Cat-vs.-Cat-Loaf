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
Convolutional Neural Networks (CNNs)
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
TABLE I
SUMMARY OF CNN COMPONENTS AND THEIR ROLES
Component Role in CNN
Convolutional Layers Extract spatial features such as edges, textures, and
shapes using learnable filters.
Pooling Layers Downsample feature maps to reduce dimensions and
retain significant information.
Fully Connected Layers Interpret extracted features and map them to class prob-
abilities for predictions.
Activation Functions Introduce non-linearity to model complex patterns (e.g.,
ReLU, Sigmoid).
Dropout Mitigate overfitting by randomly deactivating neurons
during training.
C. Hierarchical Feature Learning
CNNs are particularly effective because of their ability to
learn hierarchical features:
• Low-Level Features: The initial layers capture basic
elements such as edges, corners, and textures.
• Mid-Level Features: Intermediate layers detect more
complex patterns, such as shapes or motifs.
• High-Level Features: The deeper layers identify task-
specific features, such as objects or postures, which are
crucial for classification tasks.
This hierarchical approach enables CNNs to generalize well
across various image classification tasks, including nuanced
ones like the ”Cat vs. Cat Loaf” problem.
D. Advantages of CNNs
CNNs offer several advantages over traditional machine
learning techniques:
• Automatic Feature Extraction: Unlike traditional ap-
proaches requiring handcrafted features, CNNs learn fea-
tures directly from data.
• Parameter Sharing: Convolutional layers reuse the same
kernel weights across the input, significantly reducing
the number of trainable parameters compared to fully
connected networks.
• Spatial Invariance: Pooling layers make CNNs robust to
minor translations, rotations, and distortions in the input
image.
• Scalability: CNNs perform well on large-scale datasets
and are highly adaptable to complex tasks like object
detection and segmentation.
E. Relevance to This Project
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
III. DATASET OVERVIEW
A. Dataset Description
The ”Cat vs. Cat Loaf” dataset consists of 646 RGB images,
evenly divided into two classes:
• Cat: 323 images of cats in traditional poses.
• Cat Loaf: 323 images of cats in a loaf posture, charac-
terized by a compact body with paws tucked underneath.
Images were resized to 100 × 100 pixels to standardize
input dimensions for the CNN. This uniformity simplifies the
learning process and ensures that the model focuses on feature
extraction rather than data variability.
B. Relevance of the Dataset
This dataset presents a unique challenge in computer vision,
as the classification task relies on detecting subtle differences
in posture rather than entirely different objects. The problem
highlights the importance of spatial feature extraction in neural
networks and serves as a testbed for evaluating the robustness
of CNN architectures in nuanced classification tasks.
IV. METHODOLOGY
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
VI. RESULTS AND DISCUSSION
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
VII. CONCEPT TO CODE
The implementation of this project, including all scripts
for preprocessing, training, evaluation, and visualization, is
available in the following GitHub repository:
https://github.com/MHC-FA24-CS341CV/beyond-
the-pixels-emerging-computer-vision-research-topics-
fa24/blob/main/Imageclassif ication.ipynb
This repository contains a Jupyter Notebook detailing the
complete pipeline for ”Cat vs. Cat Loaf” image classification.
The notebook is modular and well-documented, making it
accessible for replication and further exploration.
This project’s code runs in google collab, and here is code
includes:
• Data Preprocessing: Scripts for resizing, normalizing,
augmenting, and splitting the dataset into training and
validation sets.
• Model Architecture: A Python script defining the Con-
volutional Neural Network (CNN) model used for the
classification task.
• Training and Evaluation: Scripts for training the model,
visualizing performance metrics, and saving the trained
model.
• Execution Instructions: A detailed text file outlining
how to run the code step-by-step, including required
dependencies and configurations.
A. How to Run the Code
To replicate the results and use the provided code, follow
these steps:
1) **Set Up the Environment**:
• Install Python (version 3.8 or later) and necessary
libraries such as TensorFlow, NumPy, Matplotlib,
and Pillow.
2) ** add the Dataset from kaggle**:
• The dataset is available at the repository or via the
Kaggle dataset link provided in the documentation.
3) **Run the Preprocessing Script**:
• Execute the preprocessing script to resize images to
100 × 100 pixels, normalize pixel values, and split
the data into training and validation sets.
4) **Train the Model**:
• Use the script to train the CNN on the preprocessed
dataset.
• Training progress, including accuracy and loss met-
rics, will be logged and visualized during execution.
5) **Evaluate the Model**:
• Evaluate the trained model on validation data using
the following command:
• This will output key performance metrics, such as
validation accuracy, precision, recall, and F1-score,
as well as a confusion matrix.
6) **Visualize Results**:
• Visualizations of training and validation accu-
racy/loss curves can be generated by running the
script.
7) **Save and Use the Model**:
• The trained model is saved in .h5 format for future
inference.
• Use this model to make predictions on new data by
running the script.
B. Next Steps for Technical Implementation
While the current implementation demonstrates reasonable
performance, there remain open challenges and areas for
improvement in this subfield of posture-based image classifica-
tion. These challenges include limitations in dataset diversity,
feature extraction, and model robustness. The following steps
aim to address these limitations and refine the proposed
research approach:
• **Expanding the Dataset**: One of the primary chal-
lenges in this project is the limited dataset size, which
may restrict the model’s ability to generalize to unseen
images. Adding more images representing diverse cat
postures and loaf-like poses is essential. Additionally,
sourcing images from varied backgrounds and lighting
conditions can enhance the model’s ability to handle real-
world scenarios.
• **Using Transfer Learning**: Transfer learning, lever-
aging pre-trained models like VGG16 or ResNet [6], can
address the challenge of learning robust features from a
small dataset. These models are trained on large datasets
like ImageNet and are capable of extracting hierarchical
features effectively. Fine-tuning such models on the ”Cat
vs. Cat Loaf” dataset could improve classification accu-
racy and reduce training time.
• **Incorporating Advanced Augmentations**: Data aug-
mentation techniques currently applied (e.g., rotation,
flipping, and shifting) could be extended to include more
complex transformations such as random occlusions,
color jittering, and perspective warping. These advanced
augmentations simulate diverse image variations, improv-
ing model robustness and mitigating overfitting.
• **Fine-Tuning Hyperparameters**: Optimal hyperpa-
rameter tuning is essential for achieving peak model
performance. Future steps include systematic exploration
of learning rates, batch sizes, number of layers, and
kernel sizes using techniques like grid search or Bayesian
optimization. This step addresses the limitations of man-
ual tuning by identifying configurations that maximize
validation accuracy and minimize loss.
• **Addressing Class Imbalance and Subtlety of Pos-
tures**: One open challenge in this domain is the subtlety
of differences between ”Cat” and ”Cat Loaf” classes,
which can lead to classification errors. Techniques like
focal loss, which focuses on hard-to-classify samples,
and class-specific data augmentation can be utilized to
overcome these challenges.
• **Exploring Attention Mechanisms**: Incorporating at-
tention mechanisms, such as the attention blocks used in
Vision Transformers (ViT) [7], can help the model focus
on critical regions of the image, such as the body posture
and tucked-in paws. This approach addresses the current
limitation of convolutional operations in capturing global
context effectively.
• **Research Idea**: Building on the background pro-
vided, a unique aspect of the proposed implementation
is the combination of hierarchical feature extraction (via
CNNs) with attention mechanisms to enhance focus on
subtle visual distinctions. By integrating transfer learn-
ing, advanced augmentations, and attention modules, this
approach aims to improve classification performance be-
yond conventional CNN methods.
