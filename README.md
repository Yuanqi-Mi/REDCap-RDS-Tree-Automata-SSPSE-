# REDCap-RDS-Tree-Automata-SSPSE-
A open access Streamlit-based tool for visualizing REDCap RDS recruitment trees, generating site-level Gile’s SS weights and SS-PSE population size estimates.
        

**Author:** Yuanqi Mi  
**Python version:** 3.10+  
**App type:** Streamlit Web Application  
**Updated:** 2025-11  

---

# Requirements / Dependency Versions

Python 3.10+

R 4.1+

Required R packages: RDS, sspse

Required Python libraries: streamlit, numpy, pandas, plotly, networkx, reportlab, requests




## RDS Methodology Documentation

For users who want to understand the theoretical foundations behind Respondent-Driven Sampling (RDS), the following resources provide the key statistical background and methodological guidance:



### Core RDS Statistical Review
- **[Respondent-Driven Sampling: Theory and Methods](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-031017-100704)**  
  *Annual Review of Statistics and Its Application (Gile & Handcock, 2017)*

### Official RDS R Package
The **RDS R package** provides tools for RDS-II, Gile’s SS estimators, variance estimation, and diagnostics.

- **[Package PDF Manual](https://cran.r-project.org/web/packages/RDS/RDS.pdf)**
- **[CRAN Package Index](https://cran.r-project.org/web/packages/RDS/index.html)**

---

## Official SS-PSE R Package

This application uses the Successive Sampling Population Size Estimator (SS-PSE) developed by the  
Hidden Population Methods Research Group (HPMRG).

- **[SS-PSE Package Homepage](https://hpmrg.org/sspse/)**  



The SS-PSE package includes:

- Bayesian population size estimation  
- Prior vs posterior model fit diagnostics  
- Visibility distribution plots  
- Tools for assessing model stability & sensitivity  
- Full methodological documentation and reproducible examples  

We recommend reviewing the official documentation to ensure appropriate interpretation of the posterior size estimates.


## Overview

**REDCap RDS Tree Automata** is an interactive Streamlit-based application designed to:

- Construct **Respondent-Driven Sampling (RDS)** recruitment trees  
- Visualize trees using a **Layered / Tidy Wave** layout  
- Clean and adjust **network size (degree)** reports  
- Compute **Gile’s Successive Sampling (SS) Weights** (site-level only)  
- Estimate hidden population sizes using **Successive Sampling Population Size Estimator (SS-PSE)**  
- Generate a **full PDF research report**  
- Provide an **interpretation guide** for good vs. poor SS-PSE model fits  

This app works with both **REDCap API** imports and **uploaded CSV/XLSX files**.

---

## Key Features

### 1. Recruitment Tree Construction (Tidy Wave Layout)
- Auto-detect errors in in-coupons, seeds, out-coupons, and network size 
- Quality **layered tidy** visualization  
- Optional jitter to reduce node overlap


Below is an animated demonstration of the interactive tidy-wave recruitment tree
rendered in the Streamlit app:
![Recruitment Tree Display](recruitment_tree.gif)


### 2. Network Size Cleaning Options
- Display network size distribution in the sample
- Fix underreported network sizes  
- Impute `NA` and `0` using self-imputed median
- Cap extreme degree values  
- Export cleaned degree data  

### 3. Site-Level Analysis
- Split trees by site prefix  
- Compute **Gile’s SS Weights** 
- Compute **SS-PSE**  
- Generate posterior distributions plots

![SSPSE Display](sspse.pdf)

### 4. PDF Research Report (ReportLab)
- Includes:
  - Methods  
  - Full tree summary  
  - Site-level summaries  
  - Network size cleaning diagnostics  
  - Gile’s SS weights summaries  
  - SS-PSE posterior tables and plots  
- Completely automated  

### 5. Model Fit Interpretation Tab
Includes examples of **Good Fit** vs **Poor Fit** SS-PSE posterior curves, explaining how to evaluate prior population size imputed.

---

## 🔧 Quick Start Guide

### 1️⃣ Installation

#### Clone the repository
```bash
git clone https://github.com/Yuanqi-Mi/REDCap-RDS-Tree-Automata-SSPSE.git
cd REDCap-RDS-Tree-Automata-SSPSE


#### Install Python dependencies: requires Python 3.10+.
```bash
pip install -r requirements.txt
```
#### Install R Dependencies

This app relies on two R scripts:

compute_rds_weights.R

compute_sspse.R

Install required R packages:

```r
install.packages(c("RDS", "sspse"))
```

macOS users: ensure Rscript is on your PATH:
```bash
which Rscript
```
Windows users: modify the path to Rscript.exe in app7.py:
```python
def get_rscript_path():
    return r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"
```

### 2️⃣ Launch the APP

Run Streamlit:
```bash
streamlit run app7.py
```

Your browser will automatically open at:
http://localhost:8501/
