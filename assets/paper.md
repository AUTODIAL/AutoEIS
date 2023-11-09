---
title: "AutoEIS: An automated tool for analysis of electrochemical impedance spectroscopy using evolutionary algorithms and Bayesian inference"
tags:
    - python
    - julia
    - material science
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
date: 9 November 2023
bibliography: paper.bib
---
# Summary

AutoEIS is an innovative software tool designed to revolutionize the analysis of Electrochemical Impedance Spectroscopy (EIS) data, a key technique in electrochemical research. It leverages advanced computational methods, including evolutionary algorithms and Bayesian inference, to automate the process of constructing and evaluating equivalent circuit models (ECMs). This automation provides a significant advancement in the field, allowing for more objective, efficient, and accurate analysis of EIS data compared to traditional manual methods.

In EIS, interpreting the impedance data to understand the underlying electrochemical processes is crucial. However, this interpretation often involves selecting an appropriate ECM, a task that can be complex and subjective. AutoEIS addresses this challenge by providing a systematic approach to ECM selection. It generates a wide array of potential ECMs, evaluates their fit to the EIS data, and ranks them based on statistical metrics. This process not only streamlines EIS data analysis but also introduces a level of precision and repeatability that manual methods struggle to achieve.

AutoEIS's capabilities were demonstrated through several case studies, including the analysis of oxygen evolution reaction electrocatalysis, corrosion of multi-principal element alloys, and CO2 reduction in electrolyzer devices. These studies highlighted the tool's versatility in handling different electrochemical systems and its effectiveness in identifying ECMs that accurately reflect the electrochemical processes under study.

How to write:

- Equation (inline): $\tau$ or $D_{\text{eff}}$
- Equation (block):

\begin{equation}\label{eq:tort}
D_{\text{eff}} = D\frac{\epsilon}{\tau}
\end{equation}

- Cite: [@cooper2016taufactor]
- Figure: ![Caption with a \href{https://google.com}{link} and citation [@zhang2023].\label{example}](example.pdf)
- Link: \href{https://uk.mathworks.com/matlabcentral/fileexchange/57956-taufactor}{here}
- Cite a figure: \autoref{example}

# Statement of need

Electrochemical Impedance Spectroscopy (EIS) is a critical technique in various areas of electrochemistry, including battery research, fuel cell development, and corrosion studies. The interpretation of EIS data is pivotal for understanding the mechanisms of electrochemical reactions and material behaviors. However, this interpretation is often challenging due to the complexity of the data and the requirement of expert knowledge in constructing and evaluating ECMs. This complexity can lead to significant time investment and potential bias in EIS analysis.

AutoEIS addresses this gap by providing an automated, user-friendly platform for EIS analysis that does not require extensive prior knowledge of the underlying electrochemical processes. This makes EIS analysis more accessible to a broader range of researchers and professionals in the field. By automating the ECM construction and evaluation process, AutoEIS significantly reduces the time and effort required for EIS data analysis. It also minimizes the subjectivity inherent in manual ECM selection, leading to more reliable and reproducible results.

The need for such a tool is evident in the growing complexity of electrochemical systems being studied today. As researchers explore new materials and reaction mechanisms, the ability to quickly and accurately analyze EIS data becomes increasingly important. AutoEIS meets this need by offering a high-throughput, versatile tool that can adapt to a wide range of electrochemical systems, making it an essential resource for advancing research and development in electrochemistry and related fields.

# Software Description

AutoEIS comprises four main components:

1. Data Pre-processing: It applies techniques like Kramer-Kronig transformations for initial data assessment, ensuring the reliability of EIS data for further analysis.

2. Evolutionary Algorithm-Based ECM Generation: AutoEIS generates a range of ECMs, exploring various configurations to fit the given EIS data.

3. Post-filtering of ECMs: It applies filters based on electrochemical theory to refine the ECM pool, focusing on models that are physically plausible.

4. Bayesian Inference for Model Evaluation: This step involves statistical evaluation of ECMs against EIS data, determining the most probable models using metrics like the Mean Squared Error (MSE) and the Bayesian Information Criterion (BIC).

# Authorship Contributions

RZ wrote the original AutoEIS software. MS did a major refactor of the entire code base according to best practices, also optimized performance-critical parts of the project leading to a ~10x speed-up compared to the base version, also added unit tests, documentation, and automated test and deployment workflows. The project was supervised by JH. All authors contributed to the writing and editing of the manuscript.

# Acknowledgements

We extend our thanks to Dr. Robert Black, Dr. Debashish Sur, Dr. Parisa Karimi, Dr. Brian DeCost, Dr. Kangming Li, and Prof. John R. Scully for their insightful guidance and support during the development of AutoEIS. Our gratitude also goes to Dr. Shijing Sun, Prof. Keryn Lian, Dr. Alvin Virya, Dr. Austin McDannald, Dr. Fuzhan Rahmanian, and Prof. Helge Stein for their valuable feedback and engaging technical discussions. We particularly acknowledge Prof. John R. Scully and Dr. Debashish Sur for allowing the use of their corrosion data as a key example in our work, significantly aiding in the demonstration and improvement of AutoEIS.

# References
