# Iris DL4J Classifier

This project demonstrates how to train a simple neural network using **DeepLearning4j** to classify Iris flowers. The repository contains the Java sources, a Maven configuration and example output files for the trained model and its predictions.

## Objectives

- Train a multilayer perceptron on the Iris dataset.
- Save the resulting network so it can be loaded without retraining.
- Use the saved model to generate predictions for each record in the dataset.

## Prerequisites

- **Java 17** or later.
- **Maven** to build and run the application.

Ensure your environment has enough memory to download DL4J dependencies on the first build.

## Building the project

Run the following command from the project root to resolve dependencies and compile the sources:

```bash
mvn package
```

## Training the model

Executing the `IrisClassifier` class will train a new network when no saved model is found in `modelli/irisModel.zip`:

```bash
java -cp target/iris-dl4j-1.0-SNAPSHOT.jar iris.IrisClassifier
```

After training, the model is stored under `modelli/` and reused on subsequent runs.

## Generating predictions

Running the same command again loads the saved model and writes predictions to `predizioni/output.csv`:

```bash
java -cp target/iris-dl4j-1.0-SNAPSHOT.jar iris.IrisClassifier
```

The console will show each example and the predicted class, while the CSV file contains the inputs and their predicted label.

---
