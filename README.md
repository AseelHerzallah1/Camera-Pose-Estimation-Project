# Image Processing Project – Camera Pose Estimation  

This project explores **camera pose estimation** using **3D model rendering** and **deep learning**. The goal is to predict a camera’s orientation relative to a 3D object by training a CNN on synthetic images generated from multiple viewpoints.  

---

## 📌 Project Overview  

**General Description:**  
Developed a system that predicts camera pose from 2D images of a 3D object. The approach combines **synthetic dataset generation** (via 3D rendering) with a **convolutional neural network** trained to estimate viewing angles.  

**Key Features:**  
- **Synthetic Dataset Creation:** Generated labeled images from a 3D model (kid’s slide) at different angles using PyVista & Open3D.  
- **CNN Model Training:** Implemented a Keras/TensorFlow CNN for pose classification.  
- **Evaluation:** Achieved **97% validation accuracy**; optimized dataset with denser viewpoints at edge/mid angles to reduce error.  
- **Angle Prediction:** Model outputs estimated camera angles for unseen images.  
- **Visualization:** Plotted training performance and error trends using Matplotlib.  

**Technologies:**  
`Python` · `TensorFlow/Keras` · `Open3D` · `PyVista` · `Matplotlib` · `PIL`  

---

## 🎯 Results  

- **Validation Accuracy:** 97%  
- **Error Analysis:** Minimal errors after dataset optimization; critical errors only at extreme viewpoints.  
- **Applications:** Useful for computer vision tasks such as **3D reconstruction, AR/VR, and robotics navigation**.  
