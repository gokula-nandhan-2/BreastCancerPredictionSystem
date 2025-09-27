# 🩺 Breast Cancer Prediction System  

## 📌 Overview  
This project was developed as part of the **Machine Learning and Data Mining (5DATA002W.2)** module at the University of Westminster.  

The **Breast Cancer Prediction System** applies machine learning algorithms to predict the likelihood of breast cancer based on diagnostic features. The project was implemented in **Google Colab Notebooks** and includes preprocessing, model training, evaluation, and reflection.  

---

## 🎯 Objectives  
- Explore and preprocess the dataset.  
- Apply multiple classification algorithms to build predictive models.  
- Evaluate performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.  
- Compare results and provide analysis in the coursework report.  

---

## 📂 Repository Structure  

```bash
breast-cancer-prediction-system/
│── README.md # Project documentation
│── 5DATA002W.2_Coursework_Dataset_25012025v6.0.csv # Coursework dataset
│── W2082091_20233027_Final_Python_1.ipynb # Notebook 1 (Data preprocessing & EDA)
│── W2082091_20233027_Final_Python_2.ipynb # Notebook 2 (Model training & evaluation)
│── W2082091_20233027_Final_Python_3.ipynb # Notebook 3 (Comparisons & results)
│── W2082091_20233027_ML_Coursework_Report.pdf # Final coursework report
```

---

## 📊 Dataset  
- **File:** `5DATA002W.2_Coursework_Dataset_25012025v6.0.csv`  
- **Format:** CSV (Comma-Separated Values)  
- **Rows:** 4,024  
- **Columns:** 18  
- **Type:** Breast cancer diagnostic and clinical dataset  

### 📝 Features  
1. **Patient_ID** – Unique identifier for each patient  
2. **Month_of_Birth** – Patient’s month of birth (numeric)  
3. **Age** – Age of the patient (years)  
4. **Sex** – Patient’s gender  
5. **Occupation** – Profession of the patient  
6. **T_Stage** – Tumor size/extent (T1, T2, T3…)  
7. **N_Stage** – Lymph node involvement (N1, N2, N3…)  
8. **6th_Stage** – Stage according to AJCC 6th edition  
9. **Differentiated** – Tumor differentiation (well, moderate, poorly)  
10. **Grade** – Tumor grade (1–3)  
11. **A_Stage** – Anatomical stage  
12. **Tumor_Size** – Size of tumor (mm)  
13. **Estrogen_Status** – ER status (Positive / Negative)  
14. **Progesterone_Status** – PR status (Positive / Negative)  
15. **Regional_Node_Examined** – Number of regional nodes examined  
16. **Reginol_Node_Positive** – Number of regional nodes positive  

### 🎯 Target Variables  
- **Survival_Months** – Patient survival time in months (**Regression task**)  
- **Mortality_Status** – Whether the patient is **Alive** or **Dead** (**Classification task**)  


---

## ⚙️ Requirements  
The project was developed in **Google Colab**. No manual setup is needed if run there.  

If running locally:  
- Python 3.9+  
- Jupyter Notebook / JupyterLab  
- Install dependencies:  
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn

---

## 🚀 How to Run

### Option 1 – Google Colab (Recommended)

Upload the .ipynb notebook(s) to Google Colab.
Upload the dataset file (5DATA002W.2 Coursework Dataset (25012025v6.0)).
Run all cells in sequence.

### Option 2 – Local Execution

Clone this repository.
Place the dataset file inside the data/ folder.
Open the desired notebook in Jupyter Notebook:
```bash
jupyter notebook W2082091_20233027_Final_Python_1.ipynb
```
Run all cells.

---
 
## 📊 Models Used

- Logistic Regression
- Decision Trees
- k-Nearest Neighbors (kNN)
- Naïve Bayes
- Support Vector Machines (SVM)
- (Optional: Ensemble methods if included)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve & AUC (if implemented)

---

## 📖 Learning Outcomes

- Practical experience in classification algorithms for medical datasets.- 
- Improved understanding of preprocessing, feature engineering, and model evaluation.
- Hands-on work in Google Colab for reproducible machine learning workflows.
- Critical analysis of results documented in the coursework report.
