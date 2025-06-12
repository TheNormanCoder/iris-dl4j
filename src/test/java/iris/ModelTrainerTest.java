import iris.ModelTrainer;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ModelTrainerTest {

    @Test
    public void trainedModelShouldReachMinimumAccuracy() {
        ModelTrainer trainer = new ModelTrainer();
        MultiLayerNetwork model = trainer.trainModel();

        DataSetIterator testIter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(testIter);
        testIter.setPreProcessor(normalizer);

        Evaluation eval = model.evaluate(testIter);
        double accuracy = eval.accuracy();
        assertTrue(accuracy > 0.9, "Model accuracy should exceed 0.9");
    }
}

