# Brain Tumor Segmentation with Attention U-Net

本專案使用實現 **Attention U-Net** 架構實作 MRI 影像中的腦瘤分割任務。目的是區分影像中的腫瘤區域，協助醫療影像分析自動化。

---

## 專案目標

- 使用 MRI 影像對腦瘤進行影像分割。
- 採用 Attention U-Net 模型提升對邊界與小型腫瘤區域的識別能力。
- 評估模型效能（Dice score指標）。

---

## 資料集

- 資料集來源為Mateusz Budan於kaggle所提供的[Brain MRI segmentation資料集](<https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation> "Title")。該資料為The Cancer Genome Atlas (TCGA)所收集的110位膠質瘤患者腦部影像及腫瘤標記遮罩，腦部影像使用FLAIR序列收取水平方向不同切面位置的影像，並由相關專業人員手動標記腫瘤位置，共計3929組影像，影像檔案格式皆為.tif。data/lgg-mri-segmentation中的資料夾分別以患者的醫療單位及ID命名。
|                         MR image                        |                             Mask                             |
|:--:|:--:|
|<img src="data/train/images/TCGA_DU_5851_19950428_1.tif">|<img src="data/train/images/TCGA_DU_5851_19950428_1_mask.tif">|
    


- 本專案將該資料集劃分為三個部分，分別是訓練資料集、驗證資料集以及測試資料集，各資料集的影像數量如下表所示。
| Training    | Validation  | Testing       |
|     :---:   |    :----:   |      :---:    |
| 3329        | 247         | 353           |

---

## 模型架構: Attention U-net
本專案使用[Attention U-net](<https://arxiv.org/abs/1804.03999> "Title")實作腦瘤分割模型。該模型改良自傳統 U-Net 架構，引入了 Attention Gate 模組，有效聚焦於與目標相關的特徵區域。
Attention U-net架構包含：
- Encoder（Downsampling）：多層卷積與池化擷取特徵。
- Decoder（Upsampling）：反卷積與跳躍連接(skip connection)重建影像。
- Attention Gate：於跳躍連接時選擇性地傳遞重要特徵。

其模型的架構與參數設定示意圖如下:
![image1](/models/model structure.jpg "model structure")

---

## 專案成果
訓練完的模型透過測試資料集評估模型的結果，計算Dice score、Precision和Recall的平均值做為評估模型的指標，其數值分別如下表。
| Dice Score  | Precision   | Recall        |
|     :---:   |    :----:   |      :---:    |
| 0.7629      | 0.8789      | 0.8345        |

模型的評估結果並不理想，實際觀察模型預測分割的結果後，歸類出容易產生誤判的幾個問題。下列以實際的案例進行說明。圖中左側為原始影像，中間為手動圈選的腫瘤標記遮罩。右側則為預測結果與手動圈選標記的疊圖，其中灰色區域為手動圈選標記，黃色區域為模型預測結果。
- 發現模型對於腫瘤中信號較強的區域能夠準確辨別，但較難辨別腫瘤邊界影像信號強度較弱的區域。
![image2](/results/overlay/TCGA_CS_5395_19981004_11_overlay.jpg "prediction example1")
- 模型容易將影像信號強度較強的非腫瘤區域誤判為腫瘤。
![image3](/results/overlay/TCGA_CS_4943_20000902_13_overlay.jpg "prediction example2")





