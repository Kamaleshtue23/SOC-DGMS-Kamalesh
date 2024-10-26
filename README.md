# DCGAN for MNIST Image Generation
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) on the MNIST dataset, generating images of handwritten digits. The model explores both a baseline and an enhanced architecture, comparing performance based on layer complexity, image quality, and training stability.

## About the Project
This project involves training a DCGAN to generate images of digits from the MNIST dataset, which contains grayscale images of handwritten digits. Two model versions, with varying layer complexities, are trained and compared based on their performance metrics and generated image quality.

## Dataset
The MNIST dataset is preloaded using TensorFlow and preprocessed to fit the DCGAN architecture. The dataset is normalized to the range [-1, 1], a typical requirement for GANs.

Image Size: 28x28 pixels, single-channel (grayscale)
Classes: Digits from 0-9
## Models
### Non-Enhanced Model
This baseline model has fewer layers and parameters, focusing on a simpler architecture for faster training and foundational GAN characteristics.

* Generator: Contains three transposed convolutional layers.
* Discriminator: Contains two convolutional layers with leaky ReLU activations.
### Enhanced Model
The enhanced model builds upon the baseline by adding layers to both the generator and discriminator, aiming to improve the quality and diversity of generated images.

* Generator: Additional convolutional layers improve image detail and clarity.
* Discriminator: Additional layers help in better distinguishing real vs. generated images, reducing artifacts in generated images.
## Training
The model training procedure includes:

* Loss Functions: Binary cross-entropy for both generator and discriminator losses.
* Optimizers: Adam optimizer with a learning rate of 1e-4.
* Noise Dimension: 100-dimensional noise input for the generator.
* Epochs: 50 epochs of training.
### Training Command
To train the model, run:

```
python train_dcgan.py
```
The training loop monitors generator and discriminator losses for stability and model performance.

## Comparison
The project compares the Non-Enhanced and Enhanced models based on:

### Similarities
* Basic Structure: Both models are GANs using similar architecture with a generator and discriminator.
* Training Procedure: Identical loss functions and optimizers.
* Output Format: Both generate 28x28 grayscale images normalized to [-1, 1].
### Differences
* Model Complexity: The Enhanced model has more layers and parameters.
* Loss Curves: Enhanced model shows smoother, lower loss, indicating better stability.
* Image Quality: Enhanced model images are sharper and clearer.
* Training Speed: Non-Enhanced model trains faster, while Enhanced model takes longer but yields better results.
* Diversity and Consistency: Enhanced model generates more varied, realistic images.
## Results
Sample outputs at epoch 50 are saved as:

`with_enhancement.png` - Generated images from the Enhanced model.

`without_enhancement.png` - Generated images from the Non-Enhanced model.

## License
This project is licensed under the MIT License.
