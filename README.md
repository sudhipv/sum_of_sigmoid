### This repository contains the codes to carry out Bayesian parameter estimation and predictions for the sum of sigmoid model of infection rate.

## ABSTRACT 

A sum of sigmoid representation of infection rate inspired by a shallow single layer neural network structure,  which can be adapted to fit the complex dynamics of infection and offering a better predictive capability to other models is presented. This infection rate model is represented by a series of sigmoid basis functions added to a baseline value called as the sum of sigmoid model. Each sigmoid is parametrized by a single amplitude which controls the increase in dynamics. and the temporal instance of activation (even though additional parameter refinements are possible). This sum of sigmoid model is inspired by a single hidden layer shallow neural network with specific weights and biases. The multiple sigmoids in the model are activated at specific interest points in the infection curve which are time instances at which a change in slope is detected. The underlying assumption is that this change in slope is due to any external interventions/activity which has led to a drop or spike in infection rate modelled by sigmoids. Depending upon the nature of varying infections in multiple regions, this model can be parametrized with an adaptable set of sigmoid functions. These parameters which have a nonlinear dependance to the infection data can be calibrated using infection data alone (addition of deceased cases are possible but not necessary). We demonstrate through numerical experiments that this modelling approach is applicable to multiple infection data, scalable, can quickly adapt to the evolving dynamics of infections and offers better predictive performance in comparison to other models. Comparisons to state-of-the art Wiener process model for the forecast of infections in Toronto using multiple synthetic and public health data case demonstrates the effectiveness of the method. 


<img width="1263" alt="toronto" src="https://github.com/user-attachments/assets/98a40e26-a1cf-4e57-bc9a-30ca98d4103e" />

### REPO STRUCTURE

1. PHU_data : Infection and population data for all the PHUs
2. data : All the data generated from codes required to successfully run the MCMC and predictions
3. figs : All the figures generated from different codes located here
4. inference : main source code for manual tuning, mle and mcmc




