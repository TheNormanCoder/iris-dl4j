package iris;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class IrisClassifier {
    public static void main(String[] args) {
        ModelTrainer trainer = new ModelTrainer();
        MultiLayerNetwork model = trainer.trainModel();

        ModelPredictor.predict(model);
    }
}
