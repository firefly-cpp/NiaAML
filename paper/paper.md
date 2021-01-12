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

The growing interest and development of Machine Learning (ML) methods has led to the search for ways to use these methods in the simplest way, as in general it is considered that data preprocessing and selection of appropriate ML algorithms is a demanding and time-consuming task that requires a lot of domain-specific knowledge and the use of trial-and-error approaches. That is why Automated Machine Learning (AutoML) methods have been developed. Their purpose is to automate data preprocessing and search for ML algorithms together with their hyperparameters, in order to discover the best possible ML pipeline depending on the input data [@bookHutter; @Fister2020Continuous].

AutoML can be modeled as a continuous optimization problem with several potential optimization methods applied. Stochastic population-based nature-inspired algorithms [@yang2014; @engelbrecht2007computational] are a very popular tool for dealing with such continuous optimization problems. These algorithms are inspired mainly by the biological principles and phenomena of the behavior of various species living in nature [@fister2013brief]. Each stochastic population-based nature-inspired algorithm is composed of a population of individuals that undergo different variation operators during the evolution process that results in new populations. We developed the AutoML framework NiaAML, written fully in the Python programming language. NiaAML is based mainly on the method initially proposed in [@Fister2020Continuous]. The framework incorporates stochastic population-based nature-inspired algorithms to search for the best classification pipeline [@Fister2020Continuous].

The framework is developed in a layer style layout architecture, consisting of several components, i.e. Feature Selection algorithms, Feature Transformation algorithms and classifiers. Its task is to find a perfect combination of components with proper classifiers hyperparameters` settings to build an efficient, yet customizable classification pipeline with the help of a popular collection of nature-inspired algorithms, named NiaPy [@Vrbančič2018]. Two types of optimizations in the NiaAML follow: (1) The first to find the optimal set of components for the pipeline, and (2) The second to tune the hyperparameters. Users can choose the ML components to be included into the optimization process freely, as well as select suitable fitness functions to be used for evaluation of candidate pipelines. Input data can be in the form of numerical and categorical features, as well as missing attributes, while pipelines are exported and imported as binary files for post-hoc use. Further, they can be exported as user-friendly text files that contain all of the relevant information about the pipeline and its components. A graphical outline of the NiaAML method is presented in Fig.\autoref{fig:NiaAMLflow}.

![NiaAML flow.\label{fig:NiaAMLflow}](niaamlFlow.png)

NiaAML is different: Compared to similar Python AutoML frameworks, such as TPOT [@tpot] and auto-sklearn [@NIPS2015_11d0e628], NiaAML [@Fister2020Continuous] comes with the following benefits:
* it is fully modeled as a continuous optimization problem, which means that arbitrary stochastic nature-inspired population-based algorithms [@tzanetos2020comprehensive] can be used for solving this task (without any special modifications of their internal mechanisms),
* the search for the optimal combination of ML components and proper classifiers` hyperparameters can be conducted concurrently,
* its layer style architecture allows straightforward adding of new ML components,
* every pipeline in the output is feasible and functional in cases of the correct specified hyperparameters' domains,
* the included GUI simplifies the work for regular users.

In conclusion, NiaAML is an AutoML framework based on stochastic population-based nature-inspired algorithms. Thus, the NiaAML framework is able to find the optimal classification pipelines built on preprocessing steps and classifiers, and, simultaneously, assuring a user-friendly experience. Working with ML is also, with the help of the NiaPy framework, more easy, adaptable, expandable and customizable. Last but not least, we can assure that more of the ML components and functionalities to the NiaAML framework are yet to come.

# Statement of need

Searching for the optimal classification pipeline in ML that provides the best results for a particular classification task involves a lot of domain-specific knowledge and numerous trial-and-error approaches. For instance, several conditions must be fulfilled for an ML method, such as proper preparation and preprocessing of input data, and selection of appropriate classifiers as well as their parameters. Due to the aforementioned specifics, mostly ML experts and data scientists were able to handle such tasks in the past. Empirical evidence shows that automation of the optimal classifier selection process, feature preprocessing steps and their hyperparameters, can be dealt with by non-experts [@NIPS2015_11d0e628; @7280767; @Fister2020Continuous].  With the rise of AutoML methods, dealing with the ML also became available to non-experts from other related research areas for the purpose of developing practical ML applications, solving industry-related problems with ML, or even studying out of curiosity. In here, we see the bits of NiaAML to the ML community. Simplicity to use, extendability, adaptability and customizability are just a sample of NiaAML recognitions. The reason for its simplicity lies within the layer style architecture, which is easy to understand and allows the user to add customizable components in an easy way. An available Graphical User Interface as a separate application makes the use of NiaAML even easier.

# References
