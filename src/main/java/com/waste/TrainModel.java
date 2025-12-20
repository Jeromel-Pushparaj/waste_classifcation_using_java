package com.waste;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.inputs.InputType;

import java.io.File;
import java.util.Random;

// This is the main class for training our waste classification model.
public class TrainModel {

  public static void main(String[] args) throws Exception {

    // Define image dimensions, batch size, number of training epochs, and number of classes.
    int height = 100;
    int width = 100;
    int channels = 3; // 3 channels for RGB images.
    int batchSize = 32;
    int epochs = 5;
    int numClasses = 2; // Two classes: Organic and Recyclable.

    // Point to the directories containing our training and testing images.
    File trainData = new File("dataset/TRAIN");
    File testData = new File("dataset/TEST");

    // Automatically generate labels for our images based on their parent directory.
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    // Create FileSplits for training and testing data, with a random seed for reproducibility.
    FileSplit trainSplit = new FileSplit(trainData, new Random(123));
    FileSplit testSplit = new FileSplit(testData, new Random(123));

    // Create ImageRecordReaders to read the image data.
    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);

    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
    testRR.initialize(testSplit);

    // Create DataSetIterators to iterate through the data in batches.
    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numClasses);
    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numClasses);

    // Scale image pixel values to be between 0 and 1.
    ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);
    testIter.setPreProcessor(scaler);

    // Define the architecture of our neural network.
    MultiLayerNetwork model = new MultiLayerNetwork(
        new NeuralNetConfiguration.Builder()
            .seed(123) // for reproducibility
            .updater(new Adam(0.001)) // Adam optimizer with a learning rate of 0.001
            .list()

            // First convolutional layer with 32 filters and a 3x3 kernel.
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(channels)
                .nOut(32)
                .activation(Activation.RELU)
                .build())

            // Max pooling layer to down-sample the feature maps.
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .build())

            // A fully connected (dense) layer with 64 output neurons.
            .layer(new DenseLayer.Builder()
                .nOut(64)
                .activation(Activation.RELU)
                .build())

            // The output layer with 2 neurons (one for each class) and a softmax activation function.
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())

            // Specify the input type for the network.
            .setInputType(InputType.convolutionalFlat(height, width, channels))

            .build());
    // Initialize the model.
    model.init();
    // Print the score every 10 iterations.
    model.setListeners(new ScoreIterationListener(10));

    // Train the model for the specified number of epochs.
    for (int i = 0; i < epochs; i++) {
      model.fit(trainIter);
      System.out.println("Epoch " + (i + 1) + " completed");
    }

    // Evaluate the model on the test set.
    Evaluation eval = model.evaluate(testIter);
    System.out.println(eval.stats());

    // Save the trained model to a file.
    ModelSerializer.writeModel(model, "waste-classifier.zip", true);
    System.out.println("Model saved successfully!");
  }
}
