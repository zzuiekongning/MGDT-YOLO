# Attention-Guided Edge-Optimized Network for Real-Time Detection and Counting of Pre-Weaning Piglets

This repository contains the official source code for our paper:

**Attention-Guided Edge-Optimized Network for Real-Time Detection and Counting of Pre-Weaning Piglets in Farrowing Crates**  
*(Submitted, under review)*

## 🐷 Overview

This project aims to achieve **real-time and accurate detection and counting of pre-weaning piglets** in farrowing crate environments, addressing challenges such as object overlap, background clutter, and varying lighting.

We propose an improved YOLOv8-based architecture with multiple enhancements for edge deployment.

## 🚀 Key Features

- **Real-Time Inference**: Optimized for edge deployment using [NCNN](https://github.com/Tencent/ncnn)
- **High Accuracy Detection**: Specialized for complex livestock environments
- **YOLOv8 Enhancements**:
  - 🔧 **MSPA-C2f**: A novel attention-guided C2f module for better feature extraction
  - 🧠 **Improved GD Neck**: Enhances information flow and fusion
  - 🎯 **Optimized Detection Head**: Designed for small object density and real-time performance

## 🧠 Model Architecture

We build upon [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (version `8.0.120`) 
