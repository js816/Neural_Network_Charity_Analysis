# Neural_Network_Charity_Analysis

# Overview of the Project

Alphabet Soup is a non-profit organization that makes donations to various organizations around the world.  

The purpose of this analysis is to build, train, and optimize a deep learning neural network model.  The model’s goal is to predict which organizations use Alphabet Soup’s donations successfully.


# Results

## Data preprocessing

### Model Prediction Target

The model prediction target is the Is Successful column.  In our dataset, this column is 1 (True) for those organizations that successfully used the donations and 0 (False) for those organizations that did not use the donations successfully.

### Model Features

During the initial phases of this project, the following columns were features of the model.

* Application Type
* Affiliation
* Classification
* Use Case
* Organization
* Status
* Income Amount
* Special Considerations
* Ask Amount

Preprocessing was performed on columns with high number of unique values.  Rare values were grouped into an Other category.  Columns with text values were encoded to numeric columns.  Some of the Application Type columns are shown in the image below.

![Encoding](https://user-images.githubusercontent.com/82730954/132536607-92320b96-44bf-4b5b-b71d-1c4c7022601d.PNG)


### Other Columns

The initial dataset also contained two columns that were descriptors of the organization.  The EIN and Name columns were dropped as those were not needed for our model.


## Compiling, Training, and Evaluating the Model

In the initial phases of the project, two sequential hidden layers were used in an attempt to balance model performance and processing power.  The first hidden layer had 60 neurons and the second had 20.  The RELU activation function was initially used since it’s often a good initial choice.

The model was trained for 100 epochs.

The initial model achieved 0.7190 accuracy on the test data.

![Initial_accuracy](https://user-images.githubusercontent.com/82730954/132536692-99d8b040-e35c-4812-bfc4-332d2e836ba3.PNG)

During the optimization phase of the project, a variety of methods were used.  The final testing accuracy achieved (0.7361) fell short of the goal of 0.75 but was an increase from the initial model.

![Optimized_accuracy](https://user-images.githubusercontent.com/82730954/132536735-a3210e2d-290c-4489-86d8-a043d281ada0.PNG)

Steps that were taken that resulted in the final model include:
* Dropping the Status and Model columns
* Not binning the rare values in the Classification column into “Other”
* Increasing the neurons to 100 (first hidden layer) and 80 (second hidden layer)

Additional steps that were attempted but overall found not to improve model performance include:
* Using tanh activation function
* Increasing epochs
* Using a third hidden layer
* Dropping each of the other feature columns separately
* Not binning rare Application Type values


# Summary

While the model did not achieve the goal of 0.75 accuracy, using various optimization methods, performance was improved.

Other machine learning models may provide additional insight and possibly greater performance.  For example, using a Random Forest model might provide similar or perhaps better performance.  Using a variety of decision trees and combining their results together, the model could provide us very solid performance.  Random Forest models could also potentially be less sensitive to outliers in the dataset.

Additionally, using Random Forest to analyze the importance of the various features may give us insight into which additional columns or combinations of columns could be dropped and net better performance with our current deep learning model.

It is also suggested that KerasTuner be leveraged to find the optimal values for our neural network model.
