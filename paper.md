---
title: "AutoEIS: A Python tool for automated analysis of electrochemical impedance spectroscopy"
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
date: 16 October 2023
bibliography: paper.bib
---
# Summary

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


# Authorship Contributions

RZ wrote the original AutoEIS software. MS did a major refactor of the entire code base according to best practices, also optimized performance-critical parts of the project leading to a ~10x speed-up compared to the base version, also added unit tests, documentation, and automated test and deployment workflows. The project was supervised by JH. All authors contributed to the writing and editing of the manuscript.

# Acknowledgements

This work was supported by funding from XYZ (https://faraday.ac.uk/; EP/S003053/1, grant number FIRG003 received by SK). We acknowledge contributions from Y and Z. 

# References
