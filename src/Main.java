import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        List<SetRow> data = loadSet("perceptron.test.data.csv");
        List<SetRow> trainingData = loadSet("perceptron.data.csv");

        Perceptron p = new Perceptron(data.get(0).length());

        p.train(100,trainingData,data);


    }

    public static List<SetRow> loadSet(String path) {

        try {

            Scanner sc = new Scanner(Path.of(path));
            List<SetRow> setList = new LinkedList<>();

            while (sc.hasNextLine()) {

                String[] tmp = sc.nextLine().split(",");
                double[] tmpDouble = new double[tmp.length - 1];

                for (int i = 0; i < tmp.length - 1; i++) {
                    tmpDouble[i] = Double.parseDouble(tmp[i]);
                }

                setList.add(new SetRow(tmpDouble, tmp[tmp.length - 1]));

            }

            return setList;

        } catch (IOException e) {

            e.printStackTrace();

        }

        return null;
    }

}