# Real-Time Plant Classification Model with OpenVINO

## Overview
This repository contains the implementation of a real-time plant classification model optimized using OpenVINO. The model is based on ConvNeXT Tiny architecture and has been fine-tuned on the Plant Seedlings Classification competition dataset available on Kaggle. After submission, the model achieved a remarkable 98% accuracy. Additionally, we have augmented the dataset with real greenhouse data collected from Nile University.

## Goals
- Develop a real-time plant classification model capable of running at a good frames-per-second (FPS) rate.
- Utilize ConvNeXT Tiny architecture for efficient inference.
- Optimize the model using OpenVINO to enhance performance, achieving a target FPS of 16-20.

## Dataset
The dataset consists of images of various plant seedlings captured under different conditions. It includes diverse species such as wheat, maize, and others. Additionally, real greenhouse data collected from Nile University has been incorporated to improve model generalization.

## Model Architecture
ConvNeXT Tiny architecture has been chosen for its efficiency and effectiveness in image classification tasks. The model's lightweight design allows for fast inference without compromising accuracy.

## Training
The model has been trained on the augmented dataset using standard deep learning practices. We have employed techniques such as data augmentation, transfer learning, and fine-tuning to improve performance.

## Optimization with OpenVINO
To enhance real-time performance, the trained model has been optimized using OpenVINO. OpenVINO's optimizations leverage hardware acceleration features to accelerate inference speed. With OpenVINO optimization, the model achieves a significant boost in FPS, enabling real-time inference.

## Usage
1. **Training**: Use the provided training script to train the model on your dataset. You can fine-tune the pre-trained ConvNeXT Tiny model on your specific plant classification task.
2. **Inference**: After training, deploy the model for real-time inference. Utilize OpenVINO toolkit to optimize the model for your target hardware platform. This will ensure efficient utilization of hardware resources and maximize FPS.

## Dependencies
- Python 3.x
- PyTorch
- FastAPI
- OpenVINO
- Other dependencies as specified in the requirements.txt file

## Acknowledgments
- Kaggle for providing the Plant Seedlings Classification competition dataset.
- Nile University for providing real greenhouse data for model augmentation.
- OpenVINO community for developing and maintaining a powerful toolkit for model optimization.

For detailed instructions on training, optimization, and deployment, please refer to the documentation provided in this repository. If you have any questions or suggestions, feel free to open an issue or contact us directly.

Happy coding! 🌱🔍
