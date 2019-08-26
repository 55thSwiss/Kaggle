
## Reproducing Genetic Algorithm Generated CNC Cutting Parameters with a Neural Network
### Introduction

This notebook aims to reproduce the output data of Multi-Objective Optimization of Turning Process during Machining of AlMg1SiCu (Aluminum) Using Non-Dominated Sorted Genetic Algorithm by Rahul Dhabalea, VijayKumar S. Jattib, and T.P.Singhc. The study used a genetic algorithm (GA) to generate novel cutting parameters and predict material removal rate & surface roughness. The study can be found here for further reading but some explanation will follow, mainly concerning the differences between goals and the methodology used in creating the artificial neural network (ANN).

### Background

The GA used twenty-seven rows of input data that are a series of test cuts where a CNC machine tool was set to turn a constant diameter with a range of spindle speeds (rpm), feed rates (mm/rev), and depths of cut(mm). The results of each test cut take the form of a calculated material remove rate (mm3/min) and surface roughness (μm) measurement. Their goal was to have the GA produce cutting conditions and results which maximized material removal rate and minimized surface roughness. These two outcomes are conflicting in nature and so result in one ideal output for each scenario. In total the GA generated and ranked sixteen suggestions. Five of the sixteen results were chosen for validation of the GA results and tested on the CNC machine. The results had an average of less than five percent error from the forecast.

### This Notebook

The major difference between the study and this notebook, besides the algorithm used, is in the prediction method. The ANN will be designed to accept a desired surface roughness as an input, and generate the spindle speed, feed rate, and depth of cut as outputs. The initial twenty-seven rows will be used as training data, while the five rows used for validation in the study will be split into four rows of testing data and one row for validation. The single row kept for validation will have the lowest surface roughness value in the entire dataset. This makes for a more realistic test of the ANN because an end user would be requesting results that are better than the initial test cuts and this input data will be outside anything the ANN has been trained on which makes for a more difficult prediction.

### Details

Usually a dataset's features very widely between magnitude, units, and range. For this reason, it's important to scale the data because most machine learning models recognize patterns using Euclidian distance between any two points. There are many ways to scale data but, in this notebook, the Standard method in the Scikit-Learn library will be applied. In the initial tests the MinMax scaler was also used, with both scaled data sets used with the Talos package to tune the hyperparameters of the neural network. The best results of each were applied to the Keras model for final results on the validation data. It was found for this problem the Standard scaler far surpassed the MinMax scaler, so it has been removed from the notebook. The material removal rate data, although having a higher linear correlation to the three outputs, was causing a worse model prediction when included and was dropped from the data before training. This is a simple calculated field so there is no need in "predicting" its value anyway.

### Conclusion

This notebook was made to show the feasibility in using a neural network, made with common open source tools, to facilitate process improvements in manufacturing by replicating the successful results of a GA and adjusting the input and output parameters to create a tool useful for the shop floor.

Test cuts of this nature will likely produce a very small amounts of data as seen in the study. Generally, in machine learning a much larger amount of data is required for a robust model to be created, that shortcoming can be seen here with the model's graphed validation accuracy and loss, but it appears an acceptable level of performance is achievable if enough domain knowledge around the data being collected can be applied and it's wrangled correctly. The predicted values show a much higher level of accuracy and consistency than the model's scoring during creation.

The final model is scores between 6 and 8 average percent error across the three predictions, with the RPM and feedrate consistently closer 5% and the depth of cut ranging between 10 to 15%. The fluctuation seen between model runs is caused by weights and bias in the ANN be itialized from zero at every start. The final predicted values and percent error can be seen below the graphs at the end of the notebook.

Other cutting or result data would likely need to be used to fit a real business model, for instance replacing material removal rate with some measure of process stability like tool life or "time since adjustment", but due to the proprietary nature of manufacturing the availably of real-world data is limited.
