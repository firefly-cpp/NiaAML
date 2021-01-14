---
title: 'NiaAML: AutoML framework based on stochastic population-based nature-inspired algorithms'
tags:
  - Python
  - AutoML
  - classification
  - hyperparameter optimization
  - nature-inspired algorithms
authors:
  - name: Luka Pečnik^[Corresponding author]
    orcid: 0000-0002-3897-9774
    affiliation: "1"
  - name: Iztok Fister Jr.
    orcid: 0000-0002-6418-1272
    affiliation: "1"
affiliations:
- name: University of Maribor, Faculty of Electrical Engineering and Computer Science
  index: 1
date: 20 December 2020
bibliography: paper.bib
---

# Summary

The field of Automated Machine Learning (AutoML) has been developed to automate data preprocessing and search for optimal algorithms together with their hyperparameters in order to discover the best possible ML pipeline for an input dataset [@bookHutter]. AutoML can be modeled as a continuous optimization problem with several potential optimization methods considered. Stochastic population-based nature-inspired algorithms [@yang2014; @engelbrecht2007computational] are a popular class of tools for dealing with such continuous optimization problems. These algorithms are inspired mainly by the biological behavior of various species living in nature [@fister2013brief]. Such algorithms are composed of a population of individuals that undergo different variation operations during the evolution process which results in new populations. The Python framework we have developed, NiaAML, incorporates these stochastic algorithms to search for the most suitable classification pipeline in a dataset [@Fister2020Continuous].

The framework is developed in a layer style layout architecture consisting of several components, including feature selection algorithms, feature transformation algorithms and classifiers. Its task is to find a perfect combination of components with proper classifier hyperparameter settings to build an efficient, yet customizable classification pipeline with the help of a popular collection of nature-inspired algorithms, named NiaPy [@Vrbančič2018]. NiaAML incorporates two types of optimizations, the first involves finding the optimal set of components for the pipeline, and the second involves tuning the hyperparameters. Users can freely choose the ML components to be included into the optimization process, as well as select suitable fitness functions to be used for evaluation of candidate pipelines. Input data can be in the form of numerical and categorical features, as well as missing attributes, while pipelines are exported and imported as binary files for post-hoc use. Further, they can be exported as user-friendly text files that contain all of the relevant information about the pipeline and its components. A graphical outline of the NiaAML method is presented in \autoref{fig:NiaAMLflow}.

![NiaAML flow.\label{fig:NiaAMLflow}](niaamlFlow.png)


# Statement of need

Searching for an optimal classification pipeline in ML that provides the best results for a particular classification task involves a lot of domain-specific knowledge and numerous trial-and-error approaches. For instance, several conditions must be fulfilled to adequately apply an ML algorithm, such as proper preparation and preprocessing of input data, and selection of appropriate classifiers as well as their parameters. Due to this, mostly ML experts and data scientists have been able to handle such tasks in the past. However, empirical evidence shows that automation of the optimal classifier selection process, feature preprocessing steps and their hyperparameters, can be dealt with by non-experts as well [@NIPS2015_11d0e628; @7280767].

With the rise of AutoML methods, dealing with ML has also become available to researchers from other fields. Compared to similar Python AutoML frameworks, such as TPOT [@tpot] and auto-sklearn [@NIPS2015_11d0e628], NiaAML offers the following benefits:

1. It is fully modeled as a continuous optimization problem, which means that arbitrary stochastic nature-inspired population-based algorithms can be used for solving this task without any special modifications of their internal mechanisms.

2. The search for the optimal combination of ML components and proper classifier hyperparameters can be conducted concurrently.

3. Its layer style architecture allows the straightforward addition of new ML components.

4. Every pipeline in the output is feasible and functional in cases of the correct specified hyperparameters domains.

5. A Graphical User Interface (GUI) further simplifies the work for users without expert domain knowledge.


# References
