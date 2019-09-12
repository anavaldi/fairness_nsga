# How fair can we go? Assessing the boundaries of fairness in decision trees

## Abstract

Fair machine learning works have been focusing on the develop-ment of equitable algorithms that address discrimination of certain groups. Yet, many of these fairness-aware approaches aim to obtaina unique solution to the problem, which leads to a poor understanding of the statistical limits of bias mitigation interventions.

We present the first methodology that allows to explore thoselimits within a multi-objective framework that seeks to optimize any measure of accuracy and fairness and provides a Pareto front with the best feasible solutions. In this work, we focus our study ondecision tree classifiers since they are widely accepted in machine learning, are easy to interpret and can deal with non-numerical information naturally.

We conclude experimentally that our method can optimize deci-sion tree models to be fair without compromising accuracy, which contrasts with some preliminary works in the field. However, by guiding global optimization by the non-discriminatory objective,the learning algorithm tends to produce more complex models. We believe that our contribution will help stakeholders of sociotechnical systems to assess how far they can go by being fair, accurateand explainable.


## Experimentation

We conduct an extensive set of experiments based on 5 real-world datasets, which are widely used in the fairness literature. The solution space obtained by our approach indicates that thereexists a wide number of optimal solutions (Pareto optimal), that are characterized by not being dominated by each other. We also evaluate the boundaries between accuracy and fairness that canbe achieved on each problem, giving an empirical visualization of the limits between both measures. In addition, we assess how decision trees hyperparameters are affected by this tradeoff. Finally, a convergence analysis is also presented in order to evaluate theevolutionary approach of this methodology.

![pareto_frontier_adult](pictures/pareto_frontier_adult.png)
![pareto_frontier_german](pictures/pareto_frontier_german.png)
![pareto_frontier_propublica](pictures/pareto_frontier_propublica.png)
![pareto_frontier_propublica_violent](pictures/pareto_frontier_propublica_violent.png)
![pareto_frontier_propublica_ricci](pictures/pareto_frontier_ricci.png)

**Fig. 1:** The Pareto front between accuracy and fairness on the different validation sets. Orange dots are Pareto optimal solutions obtained on each run of the meta-learning algorithm, whereas purple dots indicate the overall Pareto front solutions, i.e., *how fair can we go*. Our methodology is effective to find a wide spread of solutions which are accurate and fair at the same time. In the case of ProPublica, the meta-learning algorithm also finds better solutions than the obtained by the COMPAS algorithm, showing that a range set of possibilities of being more fair without worsening accuracy exists.

