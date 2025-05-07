# Brain Tumor Segmentation with Attention U-Net

This project implements the **Attention U-Net** architecture to perform brain tumor segmentation in MRI images. The goal is to identify tumor regions in the images and assist in the automation of medical image analysis.

---

## Purpose

- Perform image segmentation on brain tumors using MR images.
- Adopt the Attention U-Net model to enhance the ability to identify boundaries and small tumor regions.
- Evaluate model performance using the Dice score metric.

---

## Dataset

- The dataset is sourced from the [Brain MRI segmentation dataset](<https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation> "Title") by Mateusz Budan on Kaggle.
- The dataset includes brain images and tumor segmentation masks from 110 glioma patients, collected by The Cancer Genome Atlas (TCGA). The brain images were taken using the FLAIR sequence at various horizontal slice positions and manually annotated by professionals to indicate tumor regions. There are total 3,929 image-mask pairs in .tif format.
- The folders in data/lgg-mri-segmentation are named based on the medical institution and ID of each patient.
- This project splits the dataset into three subsets: training, validation, and testing. The number of images in each subset is as show in the folloing table:

<div align="center">

Training | Validation | Testing
:------: | :------: | :------: 
   3329   |     247    |   353

</div>

---

## Model Architecture: Attention U-net
This project uses the [Attention U-net](<https://arxiv.org/abs/1804.03999> "Title") for brain tumor segmentation. This model improves upon the traditional U-Net by introducing Attention Gate modules, which effectively focus on relevant feature regions.

The Attention U-Net architecture includes：
- Encoder (Downsampling): Multiple layers of convolution and pooling to extract features.
- Decoder (Upsampling): Transposed convolutions and skip connections to reconstruct the image.
- Attention Gates: Selectively pass important features during skip connections.

An illustrative diagram of the model architecture and parameter settings is shown below:
![image1](/models/model_structure.jpg "model structure")

---

## Results
The trained model was evaluated on the test dataset using Dice score, Precision, and Recall as metrics. The average values are shown in the table below:

<div align="center">

 Dice Score | Precision | Recall 
 :------: | :------: | :------: 
   0.7424   |   0.8401  | 0.8339

</div>


The evaluation results of the model are not ideal. By observing the predicted segmentation results, several issues that often lead to misjudgment were identified. 

These are explained below with real examples. In the images: the left side is the original MRI, the middle is the manually annotated tumor mask, and the right is the overlay of the model's prediction and the manual annotation. In the overlay image, Gray areas represent manual annotations, while yellow areas are the model's predictions.

- The model accurately identifies regions with strong signals in the region of tumor, but struggles with tumor boundaries where signals are weaker.
![image2](/results/overlay/TCGA_CS_5393_19990606_6_overlay.jpg "prediction example1")
![image3](/results/overlay/TCGA_CS_5395_19981004_13_overlay.jpg "prediction example2")

- The model tends to misidentify non-tumor regions with strong signals as tumors.
![image4](/results/overlay/TCGA_CS_5393_19990606_13_overlay.jpg "prediction example3")
![image5](/results/overlay/TCGA_CS_6665_20010817_14_overlay.jpg "prediction example4")

---

## Referneces
1. https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
2. Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M.J., Heinrich, M.P., Misawa, K., Mori, K., McDonagh, S.G., Hammerla, N.Y., Kainz, B., Glocker, B., & Rueckert, D. (2018). Attention U-Net: Learning Where to Look for the Pancreas. ArXiv, abs/1804.03999.
3. Franco-Barranco, Daniel & Muñoz-Barrutia, Arrate & Arganda-Carreras, Ignacio. (2021). Stable Deep Neural Network Architectures for Mitochondria Segmentation on Electron Microscopy Volumes. Neuroinformatics. 20. 10.1007/s12021-021-09556-1. 

