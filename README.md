# YOLO NAS for Capsule Endoscopy Abnormality Detection

This repository implements YOLO NAS (Neural Architecture Search) for detecting anomalies in capsule endoscopy images. The model is designed to automatically search for an optimal architecture to identify abnormalities like bleeding, polyps, etc., in real-time from capsule endoscopy videos.
![image](https://github.com/A-dvika/SEEAI_Capsule_Endoscopy_YOLO_NAS/assets/115079077/bd0ca46b-ccfc-4079-a054-4d897c5094ef)

![image](https://github.com/A-dvika/SEEAI_Capsule_Endoscopy_YOLO_NAS/assets/115079077/eaea38d6-21c3-4264-bb03-87a5c8c35ce1)

# Capsule Endoscopy

Capsule Endoscopy (CE) is a non-invasive medical imaging technique that utilizes a small wireless camera embedded in a capsule to capture images of the digestive tract. This innovative technology allows for a thorough examination of the gastrointestinal system, aiding in the detection and diagnosis of various conditions. CE is particularly useful in visualizing areas that are challenging to reach with traditional endoscopy methods.


## Overview
This project focuses on leveraging Artificial Intelligence (AI) for bleeding detection in Capsule Endoscopy (CE) videos. The AI model is trained on a dataset comprising 2618 annotated frames, encompassing various gastrointestinal bleeding instances. The goal is to enhance the efficiency, accuracy, and accessibility of bleeding detection in the context of non-invasive medical imaging.

## Key Features
### Early Detection: 
The AI model enables early identification of gastrointestinal bleeding, facilitating timely medical intervention.

### Non-Invasive Approach: 
Capitalizing on the non-invasive nature of Capsule Endoscopy, the project offers a patient-friendly diagnostic process.

### Efficiency and Speed:
Rapid processing of large CE video datasets for quick analysis of potential bleeding instances.

### Enhanced Accuracy:
Utilizing machine learning algorithms to improve the accuracy of bleeding detection, minimizing false positives/negatives.

### Generalization and Vendor Independence:
The project aims for broad applicability, ensuring the model can be generalized across different scenarios and equipment.


### Resource Optimization: 
Automated bleeding detection allows healthcare providers to optimize resources and focus human expertise on complex cases.

## How It Helps
### Improved Patient Outcomes: 
Early detection contributes to better patient outcomes by enabling timely medical intervention.

### Efficient Diagnostic Process: 
The project enhances the efficiency of the diagnostic process through automated bleeding detection.

### Non-Invasive Diagnostic Experience: 
Patients benefit from a more comfortable and less invasive diagnostic experience compared to traditional endoscopic methods.

### Optimized Resource Allocation: 
Healthcare providers can optimize resources by automating the initial screening process, directing human expertise where it's needed most.

### Contribution to Research: 
The project's datasets contribute to ongoing research in medical imaging and AI, fostering advancements in understanding and addressing gastrointestinal conditions.

# YOLO NAS Object Detection Model
![image](https://github.com/A-dvika/SEEAI_Capsule_Endoscopy_YOLO_NAS/assets/115079077/7f3b084c-d2f8-4119-bb13-7736612d3671)

This repository contains the implementation of YOLO NAS (Neural Architecture Search) for object detection. The model is designed to automatically create a convolutional neural network architecture optimized for accurate object detection.

## Overview

- Neural Architecture Search (NAS) algorithm is employed to autonomously develop a convolutional neural network tailored for object detection.
- The search space encompasses various operations such as convolutions, shortcut connections, and pooling.
- The model consists of a dynamically designed backbone network responsible for feature extraction from input images. This backbone can vary in depths and widths.
- A feature pyramid network is utilized to generate a multi-scale feature pyramid from the backbone output, facilitating object detection across different scales.
- The detection header comprises convolutional layers and prediction layers to generate bounding boxes, class scores, etc., from the feature pyramid.
- Post-processing steps, including Non-Maximum Suppression (NMS), are applied to the predictions to generate the final detections.

## Usage

### Training
1. **NAS Search:** Run the neural architecture search algorithm using a proxy task like classification.
2. **Architecture Transfer:** Transfer the top-performing architectures from the proxy task to the full detection model.
3. **Training:** Retrain the full detection model with the selected architectures for object detection.


## Results
![image](https://github.com/A-dvika/SEEAI_Capsule_Endoscopy_YOLO_NAS/assets/115079077/270bf317-2c7a-4987-ad52-f9d08cacd6d8)


- **Training Data:** Capsule endoscopy images are used as training data for NAS to search for an optimal architecture.
- **Proxy Task:** A classification dataset is constructed using capsule images labeled with anomaly/normal tags to serve as the proxy task for NAS.
- **NAS Process:** Searches for efficient backbone CNN blocks to effectively extract visual features from capsule images.
- **Model Training:** The best architecture found by NAS is further trained for detection by incorporating detection headers and fine-tuning on bounding box labeled capsule data.
- **Output:** Provides bounding boxes around anomalies along with classification scores.
- **Data Augmentation:** Utilizes techniques like rotations, flips, and color changes during NAS and detection training for improved robustness.
- **Hardware Consideration:** Focuses on searching for efficient, smaller architectures suitable for capsule devices with limited hardware.
- **Inference:** Runs in real-time on new capsule endoscopy videos to identify anomalies.
- **Post-Processing:** Applies methods like NMS to refine and curate the final detections.
- **Advantages:** YOLO NAS offers an efficient approach, leveraging architecture search to design high-accuracy and fast anomaly detection models tailored for capsule endoscopy.

## Usage

### Training
1. **NAS Search:** Execute the neural architecture search algorithm using capsule endoscopy images and the constructed classification dataset.
2. **Architecture Transfer:** Transfer the best-performing architecture to the detection model and fine-tune on bounding box labeled capsule data.
3. **Data Augmentation:** Apply augmentation techniques during training for improved model robustness.

### Inference
1. **Model Loading:** Load the trained model.
2. **Real-time Inference:** Run the model on new capsule endoscopy videos to identify anomalies.
3. **Post-processing:** Apply NMS or other post-processing methods to refine detections.


