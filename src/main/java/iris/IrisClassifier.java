package iris;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.File;

public class IrisClassifier {
    public static void main(String[] args) {
        MultiLayerNetwork model;
        File modelFile = new File("modelli/irisModel.zip");

        if (modelFile.exists()) {
            model = ModelPredictor.loadModel();
        } else {
            ModelTrainer trainer = new ModelTrainer();
            model = trainer.trainModel();
        }

        ModelPredictor.predict(model);
    }
}
