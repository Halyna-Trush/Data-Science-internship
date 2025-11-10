# 🧠 Data Science Internship Test
# Task 1 (NLP: Named Entity Recognition)

This repository contains the **first task** from the _Data Science Internship Test_, focused on **Natural Language Processing (NLP)** and specifically on **Named Entity Recognition (NER)** for identifying _mountain names_ in text.  
It demonstrates skills in dataset creation, model fine-tuning, evaluation, and reporting — fully aligned with the official task requirements.

---

## 🎯 Test Goal

The goal of this internship test is to evaluate knowledge and practical skills in:

- **Natural Language Processing (NLP)**
- **Computer Vision (CV)**

This repository presents the complete solution for **Task 1 (NLP: NER)** —  
training a model that can automatically detect **mountain and mountain range names** in English sentences.

---

## 📘 Implementation Format

> **Note:**  
> Although the task requirements specify `.py` scripts for model training and inference, both stages are implemented as **reproducible Jupyter notebooks** —  
> `mountain_ner_training_demo.ipynb` and `inference.ipynb`.  
> This format provides full transparency, inline explanations, and allows direct visualization of intermediate results.

---

## 📑 Task Requirements (as stated in the test)

- Create or find a **dataset** with labeled mountain names.
- Select a **relevant model architecture** for NER (recommended: BERT).
- **Train / fine-tune** the model on the dataset.
- Prepare demo code or Jupyter notebook for inference results.

### Expected Output:

- ✅ Jupyter notebook explaining dataset creation
- ✅ Dataset (train, valid, test + dictionary)
- ✅ Link or file with model weights
- ✅ `*.py` — training script (implemented via notebook)
- ✅ `*.py` — inference script (implemented via notebook)
- ✅ Demo notebook with inference results
- ✅ Short report (PDF) with improvement ideas

### General requirements for the whole test:

- All code written in **Python 3**
- Code must be **clear, documented, and reproducible**
- Each task placed in a **separate folder**
- Must include a **`requirements.txt`** file with all used libraries
- Final **report** (PDF) summarizing performance and ideas for improvement

---

## 🧩 Project Structure

```
Natural-Language-Processing/
│
├── data/
│   ├── Mountain_Dictionary.txt      # Global dictionary of mountain names
│   ├── train.txt                    # Training dataset in BIO format
│   ├── valid.txt                    # Validation dataset
│   ├── test.txt                     # Test dataset
│
├── notebooks/
│   ├── mountain_ner_training_demo.ipynb   # Full training pipeline (training stage)
│   ├── inference.ipynb                    # Model inference and demo
│
├── requirements.txt                 # List of all dependencies
├── NER_Mountain_Report.pdf   # Final bilingual report (EN + UA)
└── README.md
```

> 🧩 _Note:_  
> The project does not include separate `.py` scripts because all functionality (training, evaluation, and inference)  
> is fully implemented and documented inside the Jupyter notebooks.

---

## ⚙️ Model & Dataset Details

- **Base model:** `bert-base-uncased`
- **NER labels:** `{O, B-MOUNT, I-MOUNT}`
- **Dataset size:** ~1,260 sentences (~16,000 tokens)
- **Label distribution:**
  - `O` = 88.6%
  - `B-MOUNT` = 6.3%
  - `I-MOUNT` = 5.1%

**Dataset Source:**  
A _synthetic dataset_ generated from a structured Mountain Dictionary covering all continents (including the Ukrainian Carpathians and Crimean Mountains).  
Sentences were created programmatically using linguistic templates to ensure diversity and naturalness.

---

## 🧮 Training & Inference

The model was fine-tuned using the **Hugging Face Transformers** library and evaluated on a held-out test split.  
Inference is implemented both as a notebook and as an **interactive console mode**, where users can input any sentence and receive token-level predictions.

**Example:**

```
Enter a sentence: The sunrise over Emerald Summit was breathtaking.

Tokens & Labels:
The        -> O
sunrise    -> O
over       -> O
Emerald    -> B-MOUNT
Summit     -> I-MOUNT
was        -> O
breathtaking -> O

Entities: ['Emerald Summit']
```

---

## 📊 Evaluation Results

| Split      | Precision | Recall | F1-score |
| ---------- | --------- | ------ | -------- |
| Validation | 1.0000    | 1.0000 | 1.0000   |
| Test       | 0.9565    | 0.9778 | 0.9670   |

**Label Distribution (TEST split):**
| Tag | Count | Ratio |
|------|--------|--------|
| B-MOUNT | 586 | 0.0634 |
| I-MOUNT | 469 | 0.0507 |
| O | 8188 | 0.8859 |

---

## 🧩 Example Test Sentences (Unseen Names)

To test generalization, new sentences were generated with **mountain names not present** in any dataset split.  
The model successfully identifies the bolded names as **B-MOUNT/I-MOUNT** entities.

**Examples:**

1. We finally reached the base of **Mount Crystal** after a two-day trek.
2. Heavy snow covered the slopes of **Ironpeak Ridge** last night.
3. The sunrise over **Emerald Summit** was breathtaking.
4. Climbers often train on **Mount Silverhorn** before tackling higher peaks.
5. The trail to **Blue Mist Mountains** passes through dense cedar forest.
6. From our camp we could clearly see **Stormveil Range** in the distance.
7. Local legends say that **Whispering Peak** is haunted by ancient spirits.
8. Scientists installed a new weather station on **Mount Aurora**.
9. We set up tents in a meadow below **Falconcrest Hills**.
10. The helicopter circled above **Crimson Spire** to take photos of the ridge.
11. A small path winds through the foothills of **Moonlight Highlands**.
12. They compared rock samples from **Mount Solara** and **Obsidian Ridge**.
13. The team decided to avoid climbing **Frostveil Mountain** due to high winds.
14. An old monastery lies at the foot of **Sapphire Crown Range**.
15. Hikers said the route around **Mount Graystone** was surprisingly easy.
16. From the riverbank we admired the outline of **Dragon’s Spine Mountains**.
17. Our drone footage captured a stunning panorama of **Verdant Crest**.
18. Heavy fog rolled in from **Echo Vale Hills** before sunset.
19. Local maps show a new trail connecting **Mount Ashen** to **Silverwind Ridge**.
20. Few explorers have reached the icy top of **Northlight Peak**.

---

## 💡 Insights & Reflection

This project demonstrated that even a carefully designed **synthetic dataset** can train a model that generalizes well to unseen data.  
Data quality depends not on origin but on **internal structure, consistency, and contextual realism**.  
Synthetic corpora, when engineered thoughtfully, can accelerate model prototyping — especially in **low-resource NLP domains**.

---

## 🔧 Possible Improvements

- Apply stronger **regularization** to reduce overfitting.
- Add **hard negatives** (non-geographic “mount” examples).
- Experiment with **CRF** or **RoBERTa** architectures.
- Combine **synthetic and real corpora** for improved robustness.
- Extend interface to a **web-based demo**.

---

## 🧰 Technologies Used

- **Python 3.10+**
- **Transformers (Hugging Face)**
- **PyTorch**
- **pandas, numpy, scikit-learn**
- **matplotlib**
- **Jupyter Notebook**

---

> “Mountains teach persistence — and so does data.” 🏔

# 🛰 Task 2 (Computer Vision: Sentinel-2 Image Matching)

This repository contains the **second task** from the _Data Science Internship Test_, focused on **Computer Vision (CV)** and the development of an algorithm for **matching Sentinel-2 satellite images** captured in different seasons.  
It demonstrates the use of both classical and deep-learning methods for keypoint detection, feature matching, and reproducible benchmarking.

---

## 🎯 Test Goal

The objective of this task is to evaluate skills in:

- **Satellite image preprocessing**
- **Classical computer vision algorithms**
- **Deep-learning feature matching**
- **Result visualization and performance analysis**

The implemented pipeline matches satellite scenes taken in different seasons and evaluates how well the algorithms handle lighting, color, and texture variations.

---

## 📘 Implementation Format

All functionality is implemented within a **single reproducible Jupyter notebook** —  
`Sentinel.ipynb` — which integrates algorithm design, interactive interface, and result visualization.

Although the task specification mentions separate `.py` scripts, all steps (data loading, matching, evaluation, and visualization) are implemented directly in the notebook to ensure **transparency, modularity, and reusability**.

---

## 📑 Task Requirements (as stated in the test)

- Prepare a **dataset** for keypoint detection and image matching.  
- Build or train the **algorithm**.  
- Prepare **demo code** or notebook showing inference results.

### Expected Output:

- ✅ Jupyter notebook explaining dataset preparation  
- ✅ Link to the dataset (Google Drive or Kaggle)  
- ✅ Python scripts or equivalent notebook cells for training/inference  
- ✅ Visualization of keypoints and their matches  
- ✅ Short performance report (PDF)

---

## 🧩 Project Structure

```
Computer-Vision/
│
├── Sentinel.ipynb                    # Complete notebook: algorithms, interface, and visualization
├── requirements_cv.txt               # List of dependencies
├── Sentinel_Image_Matching_Report.pdf   # Final report (EN)
```

---

## ⚙️ Algorithms Implemented

1. **ORB + RANSAC (Classical)**  
   - Detects and matches keypoints using ORB descriptors.  
   - Uses RANSAC to remove outliers and estimate homography.  
   - Computes metrics: total matches, inlier ratio, reprojection error, runtime.

2. **LoFTR (Deep Learning, Kornia)**  
   - Pretrained “outdoor” LoFTR model used for robust matching under seasonal changes.  
   - Evaluated with MAGSAC for inlier filtering.  
   - Provides significantly higher accuracy and stability across different lighting conditions.

---

## 🗺 Dataset

The experiments use **four Sentinel-2 scenes (Tile T36UYA)** representing the same region in different seasons:

- 2016-02-12 — Winter (02)  
- 2019-03-13 — Spring (03) 
- 2019-06-01 — Summer (06)  
- 2019-09-09 — Autumn (09)  

📦 Dataset source:  
**[Deforestation in Ukraine from Sentinel-2 data](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine)** on Kaggle.  

📂 **Folder with `.jp2` files (Google Drive):**  
[https://drive.google.com/drive/folders/1j7u9IsiKA4RF5gTftOiknp_pBX2NpiMk?usp=sharing](https://drive.google.com/drive/folders/1j7u9IsiKA4RF5gTftOiknp_pBX2NpiMk?usp=sharing)

To run the notebook, download `.jp2` files from the dataset and place them in:  
`/content/drive/MyDrive/Sentinel` (for Colab) or `./data/Sentinel` (for local).

---

## 📊 Evaluation Results

| Image Pair | ORB: Inliers (%) | LoFTR: Inliers (%) | ORB Time (s) | LoFTR Time (s) |
|-------------|------------------|---------------------|---------------|----------------|
| 02–03 | 1.49 | 0.98 | 0.94 | 2.33 |
| 03–06 | 1.00 | 0.99 | 0.60 | 0.79 |
| 06–09 | 1.18 | 0.99 | 0.46 | 0.86 |
| 02–09 | 2.06 | 0.98 | 0.45 | 0.83 |

---

## 💡 Key Insights

- **ORB + RANSAC** performs poorly under strong seasonal variations (≤2% inliers).  
- **LoFTR** maintains stable accuracy (0.98–0.99 inlier ratio) across all seasons.  
- **Runtime** for LoFTR is within 0.8–2.3 seconds per image pair (1024×1024 px).  
- The **interactive interface** allows easy selection and comparison of scenes, ensuring reproducibility.

---

## 🔧 Technologies Used

- **Python 3.10+**  
- **Rasterio**, **OpenCV**, **Kornia**, **PyTorch**, **NumPy**, **Matplotlib**  
- **Google Colab** for execution and Drive integration

---

## 🪪 Author

**Halyna Trush**  
📍 Junior Data Scientist | NLP & ML Enthusiast  
📫 frolova.galka@gmail.com  
🌐 [LinkedIn](https://linkedin.com/in/halyna-trush)

---

> “From pixels to patterns — seeing the seasons through data.” 🌍  

