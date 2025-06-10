package iris;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;

public class ModelPredictor {

    public static MultiLayerNetwork loadModel() {
        try {
            File modelFile = new File("modelli/irisModel.zip");
            return MultiLayerNetwork.load(modelFile, true);
        } catch (Exception e) {
            throw new RuntimeException("Errore nel caricamento del modello", e);
        }
    }

    public static void predict(MultiLayerNetwork model) {
        IrisDataSetIterator testIterator = new IrisDataSetIterator(1, 150);

        File file = new File("predizioni/output.csv");
        file.getParentFile().mkdirs();

        try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
            writer.println("input1,input2,input3,input4,predicted_class");

            int i = 1;
            while (testIterator.hasNext()) {
                DataSet ds = testIterator.next();
                INDArray features = ds.getFeatures();
                INDArray output = model.output(features);
                int predictedClass = output.argMax(1).getInt(0);

                System.out.println("Esempio " + i + " → Predizione: " + output);
                i++;

                // Prepara i valori input
                String inputValues = Arrays.toString(features.toDoubleVector())
                        .replaceAll("[\\[\\]\\s]", "");

                writer.printf("%s,%d\n", inputValues, predictedClass);
            }

            System.out.println("Predizioni salvate in: " + file.getAbsolutePath());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
