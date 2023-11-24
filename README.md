**Facial Recognition on Edge Devices - README**

**Project Abstract:**
---------------------

Welcome to the Facial Recognition on Edge Devices project! In this endeavor, we explore the integration of compact edge computing for efficient facial recognition in constrained environments, with a focus on applications like drones. Leveraging the advantages of edge devices, particularly their smaller sizes, the project aims to demonstrate the practicality of deploying facial recognition technology in scenarios where traditional computing resources are impractical.

**Methodology:**
-----------------

1. **Dataset Collection and Annotation:**
   - Gather a diverse dataset of human images.
   - Annotate faces within the dataset for training purposes.

2. **YOLO Model Training for Face Detection:**
   - Train a YOLO architecture for accurate face detection in annotated images.

3. **Face Extraction and Preprocessing:**
   - Extract and preprocess detected faces for consistency.

4. **Single Shot Learning for Facial Recognition:**
   - Utilize a single-shot learning model to compare preprocessed faces to stored reference images and assign labels.

5. **Threshold-based Classification:**
   - Implement a threshold for classification to ensure recognition only for highly similar faces, minimizing false positives.

6. **Model Optimization for Edge Deployment:**
   - Optimize the trained model for deployment on edge devices.
   - Utilize techniques like quantization and pruning to balance model size and speed.

7. **Edge Device Implementation and Throughput Enhancement:**
   - Port the optimized model to the selected edge device, considering hardware constraints.
   - Implement parallelization and concurrency strategies to maximize throughput on the edge device.
