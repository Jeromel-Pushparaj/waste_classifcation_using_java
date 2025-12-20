package com.waste;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;

public class PredictImage {

    public static void main(String[] args) throws Exception {

        int height = 100;
        int width = 100;
        int channels = 3;

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("waste-classifier.zip");

        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray image = loader.asMatrix(new File("sample.jpg"));

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        INDArray output = model.output(image);
        int predictedClass = output.argMax(1).getInt(0);

        if(predictedClass == 0)
            System.out.println("Prediction: Organic Waste");
        else
            System.out.println("Prediction: Recyclable Waste");
    }
}

