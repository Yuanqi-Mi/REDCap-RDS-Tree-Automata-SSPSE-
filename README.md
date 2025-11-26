# REDCap-RDS-Tree-Automata-SSPSE-
A Streamlit-based tool for visualizing REDCap RDS recruitment trees, cleaning and adjusting reported network sizes, and generating site-level Gile’s SS weights and SS-PSE population size estimates. Includes tidy-wave tree layouts, posterior model-fit guidance, and automated PDF reporting for research workflows.

# REDCap RDS Tree Automata (Tidy Wave Only, English UI)

**Author:** Yuanqi Mi  
**Python version:** 3.10+  
**App type:** Streamlit Web Application  
**Updated:** 2025-11  

---

## 📘 RDS Methodology Documentation

For users who want to understand the theoretical foundations behind Respondent-Driven Sampling (RDS), the following resources provide the key statistical background and methodological guidance:

### 🔹 Core RDS Statistical Review  
- **Respondent-Driven Sampling: Theory and Methods**  
  *Annual Review of Statistics and Its Application (Gile & Handcock, 2017)*  
  ➤ [https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-031017-100704](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-031017-100704)

### 🔹 Official RDS R Package  
The **`RDS` R package** provides tools for RDS-II, Gile’s SS estimators, variance estimation, and diagnostics.

- 📄 Package PDF manual:  
  ➤ [https://cran.r-project.org/web/packages/RDS/RDS.pdf](https://cran.r-project.org/web/packages/RDS/RDS.pdf)

- 📦 CRAN package index:  
  ➤ [https://cran.r-project.org/web/packages/RDS/index.html](https://cran.r-project.org/web/packages/RDS/index.html)

---

## 📦 SS-PSE R Package

This application uses the **Successive Sampling Population Size Estimator (SS-PSE)** developed by the  
**Hidden Population Methods Research Group (HPMRG)**.

- 📘 SS-PSE package homepage:  
  ➤ [https://hpmrg.org/sspse/](https://hpmrg.org/sspse/)

The SS-PSE package includes:

- Bayesian population size estimation  
- Prior vs posterior model fit diagnostics  
- Visibility distribution plots  
- Tools for assessing model stability & sensitivity  
- Full methodological documentation and reproducible examples  

We recommend reviewing the official documentation to ensure appropriate interpretation of the posterior size estimates.


## 📌 Overview

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

## 🌟 Key Features

### 🧬 1. Recruitment Tree Construction (Tidy Wave Layout)
- Auto-detect incoming coupon, seeds, out-coupons, and network size fields  
- Beautiful, publication-quality **layered tidy** visualization  
- Optional jitter to reduce node overlap  

### 🧼 2. Network Size Cleaning Options
- Fix underreported network sizes  
- Impute `NA` and `0` using the dataset median  
- Cap extreme degree values  
- Export cleaned degree data  

### 🎯 3. Site-Level Analysis
- Split trees by site prefix  
- Compute **Gile’s SS Weights** at site level  
- Compute **SS-PSE** at site level  
- Generate posterior distributions & visibility plots  

### 📊 4. PDF Research Report (ReportLab)
- Includes:
  - Methods  
  - Full tree summary  
  - Site-level summaries  
  - Network size cleaning diagnostics  
  - Gile’s SS weights summaries  
  - SS-PSE posterior tables and plots  
- Completely automated  

### 🎨 5. Model Fit Interpretation Tab
Includes examples of **Good Fit** vs **Poor Fit** SS-PSE posterior curves, explaining how to evaluate priors.

---

## 📷 Screenshots (Optional)

Add screenshots here:


