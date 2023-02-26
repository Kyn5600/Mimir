import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.Arrays;
import java.util.regex.Pattern;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import weka.classifiers.bayes.NaiveBayes;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        try {
            List<String> lines = Files.readAllLines(Paths.get("conversational_english.csv"));
            List<String[]> records = lines.stream().map(line -> line.split(",")).collect(Collectors.toList());

            Collections.shuffle(records, ThreadLocalRandom.current());

            int trainSize = (int) Math.round(records.size() * 0.8);
            int testSize = records.size() - trainSize;

            List<String[]> trainRecords = records.subList(0, trainSize);
            List<String[]> testRecords = records.subList(trainSize, records.size());

            String[] header = trainRecords.get(0);

            FastVector attributes = new FastVector();

            for (int i = 1; i < header.length; i++) {
                Attribute attribute = new Attribute(header[i]);
                attributes.addElement(attribute);
            }

            Instances trainInstances = new Instances("Train", attributes, trainRecords.size());
            trainInstances.setClassIndex(attributes.size() - 1);

            for (int i = 1; i < trainRecords.size(); i++) {
                String[] values = trainRecords.get(i);
                Instance instance = new DenseInstance(values.length);

                for (int j = 0; j < values.length; j++) {
                    instance.setValue((Attribute) attributes.elementAt(j), Double.parseDouble(values[j]));
                }

                trainInstances.add(instance);
            }

            NaiveBayes naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(trainInstances);

            ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream("conversational_english_classifier.model"));
            objectOutputStream.writeObject(naiveBayes);
            objectOutputStream.close();

            System.out.println("Classifier saved to conversational_english_classifier.model");

            while (true) {
                System.out.print("Enter a conversational text: ");
                String userInput = scanner.nextLine();

                if (userInput.equals("exit")) {
                    break;
                }

                String[] tokens = userInput.split("\\s+");
                double[] values = new double[attributes.size()];

                for (int i = 1; i < tokens.length; i++) {
                    Attribute attribute = (Attribute) attributes.elementAt(i - 1);
                    if (attribute.isNominal()) {
                        values[i - 1] = attribute.indexOfValue(tokens[i]);
                    } else {
                        values[i - 1] = Double.parseDouble(tokens[i]);
                    }
                }

                Instance instance = new DenseInstance(1.0, values);
                instance.setDataset(trainInstances);

                double prediction = naiveBayes.classifyInstance(instance);

                String predictedClass = trainInstances.classAttribute().value((int) prediction);
                System.out.println("Predicted class: " + predictedClass);
            }

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}