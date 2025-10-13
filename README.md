# 🏥 Medical Insurance Cost Prediction using Linear Regression

## 📘 Project Overview
This project aims to **predict medical insurance costs** for individuals based on their demographic and lifestyle attributes such as **age, gender, BMI, number of children, smoking status, and region**.  
By using **Linear Regression**, the model estimates an individual's likely insurance charges, helping insurance companies set fair premiums and customers understand cost drivers.

---

## 🧠 Problem Statement
Insurance premiums vary greatly depending on personal factors.  
The challenge is to build a **machine learning model** that can:
- Identify the relationship between these factors and medical costs  
- Predict the expected insurance charges for new individuals  

---

## 📊 Dataset Information
Dataset file: (https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Total Records:** 1,338  
**Features:**

| Column Name | Description |
|--------------|-------------|
| `age` | Age of the person |
| `sex` | Gender (male/female) |
| `bmi` | Body Mass Index (indicator of body fat) |
| `children` | Number of dependents covered by insurance |
| `smoker` | Whether the person smokes or not |
| `region` | Residential region (northeast, northwest, southeast, southwest) |
| `charges` | Medical insurance cost (target variable) |

---

## 🧩 Steps and Methodology

### 1. Importing Libraries
Used essential libraries such as:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
