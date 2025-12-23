# Waste Classification Java

This project is a Java-based application for classifying waste images as either organic or recyclable. It utilizes the Deeplearning4j (DL4J) library to build and train a neural network for this purpose.

## Dataset

The dataset is located in the `dataset` directory and is structured as follows:

- `dataset/TRAIN`: Contains the training images.
- `dataset/TEST`: Contains the testing images.

Within both `TRAIN` and `TEST` directories, there are two subdirectories:

- `O`: Contains images of organic waste.
- `R`: Contains images of recyclable waste.

## Prerequisites

- Java 25 or later
- Apache Maven

## Dependencies

The project uses the following main dependencies (as defined in `pom.xml`):

- `deeplearning4j-core`: Core DL4J library.
- `nd4j-native-platform`: Backend for ND4J, the numerical computing library for DL4J.
- `datavec-data-image`: For image loading and processing.
- `slf4j-simple`: A simple logging implementation.

## How to Build and Run

1.  **Build the project:**
    Use Maven to build the project and create a JAR file:
    ```bash
    mvn clean install
    ```

2.  **Run the application:**
    Once the project is built, you can run the application from the generated JAR file in the `target` directory.
    ```bash
    java -jar target/waste-classification-1.0.jar
    ```

    *(Note: The exact main class to run might need to be specified depending on the project's configuration.)*
