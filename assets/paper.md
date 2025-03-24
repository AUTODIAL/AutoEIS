---
title: "AutoEIS: A Python toolkit for automated analysis of electrochemical impedance spectroscopy and equivalent circuit modeling"
tags:
    - python
    - julia
    - materials science
    - electrochemical impedance spectroscopy
    - equivalent circuit model
    - bayesian inference
authors:
    - name: Mohammad Amin Sadeghi
      orcid: 0000-0002-6756-9117
      equal-contrib: true
      affiliation: 1 # (Multiple affiliations must be quoted)
    - name: Runze Zhang
      equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
      orcid: 0009-0004-9088-7924
      affiliation: 1
    - name: Jason Hattrick-Simpers
      corresponding: true # (This is how to denote the corresponding author)
      orcid: 0000-0003-2937-3188
      affiliation: 1
affiliations:
    - name: University of Toronto, Canada
      index: 1
date: 24 March 2025
bibliography: paper.bib
---
# Summary

AutoEIS is an innovative software tool designed to automate the analysis of Electrochemical Impedance Spectroscopy (EIS) data, a key technique in electrochemical materials research. By integrating evolutionary algorithms and Bayesian inference, AutoEIS automates the construction and evaluation of equivalent circuit models (ECM), providing more objective, efficient, and accurate analysis compared to traditional manual methods.

EIS data interpretation is fundamental for understanding electrochemical processes and generating mechanistic insights. However, selecting an appropriate ECM has historically been complex, time-consuming, and subjective [@wang2021electrochemical]. AutoEIS resolves this challenge through a systematic approach: it generates multiple candidate ECMs, evaluates their fit against experimental data, and ranks them using comprehensive statistical metrics. This methodology not only streamlines analysis but also introduces reproducibility and objectivity that manual analysis cannot consistently achieve.

The effectiveness of AutoEIS has been validated through diverse case studies, including oxygen evolution reaction electrocatalysis, corrosion of multi-principal element alloys, and CO2 reduction in electrolyzer devices [@zhang2023]. These applications demonstrate the software's versatility across different electrochemical systems and its ability to identify physically meaningful ECMs that accurately capture the underlying electrochemical phenomena.

# Statement of need

EIS is widely used in electrochemistry for applications spanning battery research, fuel cell development, and corrosion studies. Accurate interpretation of EIS data is essential for understanding electrochemical reaction mechanisms and material behaviors. Traditional EIS analysis faces three significant challenges: it requires substantial expert knowledge, consumes significant time, and introduces potential researcher bias in model selection and interpretation.

AutoEIS addresses these limitations through an automated platform that reduces the expertise barrier for rigorous EIS analysis. By systematically evaluating numerous potential circuit models, the software minimizes human bias and dramatically reduces analysis time. This automation is particularly valuable for complex systems where manual trial-and-error approaches become impractical.

Current EIS analysis tools—including open-source options like DearEIS [@yrjana2022deareis], Elchemea Analytical [@elchemea], impedance.py [@murbach2020impedance], PyEIS [@knudsen2019pyeis], and pyimpspec [@pyimpspec], as well as commercial software such as ZView, RelaxIS, and Echem Analyst—all require users to manually propose ECMs and iteratively refine them. This approach becomes increasingly unreliable as system complexity grows, as researchers may not explore the full model space or may unconsciously favor familiar circuit elements.

AutoEIS distinguishes itself by comprehensively exploring the model space through evolutionary algorithms, ensuring that potentially valuable circuit configurations are not overlooked. This capability aligns with the growing trend toward self-driving laboratories and autonomous research workflows in materials science and electrochemistry.

# Software Description

AutoEIS implements a four-stage workflow to analyze EIS data:

1. **Data Preprocessing and Validation**: Before model fitting, AutoEIS applies Kramers-Kronig transformations to validate experimental data quality. This critical step identifies measurement artifacts and ensures that only reliable data proceeds to model fitting. Poor-quality data that violates Kramers-Kronig relations is flagged, allowing researchers to address experimental issues before interpretation.
2. **ECM Generation via Evolutionary Algorithms**: AutoEIS employs evolutionary algorithms through the Julia package EquivalentCircuits.jl [@van2021practical] to generate diverse candidate ECMs. This approach efficiently explores the vast space of possible circuit configurations, including models that might not be intuitively chosen by researchers.
3. **Physics-Based Model Filtering**: The software then applies electrochemical theory-based filters to eliminate physically implausible models. For example, models lacking an ohmic resistor are automatically rejected as physically unrealistic, despite potentially good mathematical fits. This step ensures that analysis results remain consistent with established electrochemical principles.
4. **Bayesian Parameter Estimation**: For physically plausible models, AutoEIS employs Bayesian inference to estimate circuit component values and their uncertainty distributions. Unlike point estimates from traditional least-squares fitting, this approach quantifies parameter uncertainty, providing crucial information about model reliability. The Bayesian framework also enables model comparison through metrics like the Bayesian Information Criterion, helping identify the most statistically justified model complexity.

# Authorship Contributions

The original AutoEIS software was developed by RZ. MS conducted a comprehensive refactoring of the codebase that improved algorithmic efficiency. MS also implemented unit testing, expanded documentation, and established automated CI/CD workflows to ensure software reliability. JHS provided project supervision and domain expertise in electrochemical theory. All authors—RZ, MS, and JHS—contributed substantively to the writing and editing of this manuscript.

# Acknowledgements

We extend our thanks to Dr. Robert Black, Dr. Debashish Sur, Dr. Parisa Karimi, Dr. Brian DeCost, Dr. Kangming Li, and Prof. John R. Scully for their insightful guidance and support during the development of AutoEIS. Our gratitude also goes to Dr. Shijing Sun, Prof. Keryn Lian, Dr. Alvin Virya, Dr. Austin McDannald, Dr. Fuzhan Rahmanian, and Prof. Helge Stein for their valuable feedback and engaging technical discussions. We particularly acknowledge Prof. John R. Scully and Dr. Debashish Sur for allowing the use of their corrosion data as a key example in our work, significantly aiding in the demonstration and improvement of AutoEIS.

# References
