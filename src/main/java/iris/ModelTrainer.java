package iris;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class ModelTrainer {

    public MultiLayerNetwork trainModel() {
        DataSetIterator iterator = new IrisDataSetIterator(150, 150);
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(10).nOut(3)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        for (int i = 0; i < 1000; i++) {
            iterator.reset();
            model.fit(iterator);
        }

        // Salva il modello dopo l'addestramento
        try {
            File modelFile = new File("modelli/irisModel.zip");
            modelFile.getParentFile().mkdirs(); // Crea la cartella se non esiste
            model.save(modelFile, true);
            System.out.println(" Modello salvato in: " + modelFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }

        return model;
    }
}
