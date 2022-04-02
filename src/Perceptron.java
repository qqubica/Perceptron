import java.util.List;

public class Perceptron {

    private double[] weights;
    private double bias;
    private double learningRate = 0.01;
    double maxEfficiency = 0;
    int afterXIterations = 0;

    public Perceptron(int inputLength) {
        weights = new double[inputLength];
        for (int i = 0; i < inputLength; i++) {
            weights[i] = Math.random();
        }
        bias = Math.random();
    }

    public void train(int repeat, List<SetRow> traningSet, List<SetRow> data) {
        for (int i = 0; i < repeat; i++) {

            for (int j = 0; j < traningSet.size(); j++) {
                train(traningSet.get(j));
            }

            System.out.println(new StringBuilder().append("After ").append(i + 1).append(" itterations succesfuly classified ").append(calculateEfficiency(data, i+1)).append("% of data points"));

        }

        System.out.println("\nMAX efficiency after traning\t" + maxEfficiency + "%\t after " + afterXIterations + "\titteration");

    }

    private int classify(SetRow input){
        return fNet(net(input)) == input.getResult() ? 1 : 0;
    }

    private double calculateEfficiency(List<SetRow> data, int itteration) {
        double correct = 0;

        for (int i = 0; i < data.size(); i++) {
            correct += classify(data.get(i));
        }

        if (maxEfficiency < correct / data.size() * 100){
            maxEfficiency = correct / data.size() * 100;
            afterXIterations = itteration;
        }

        return correct / data.size() * 100;
    }

    private void train(SetRow input) {
        double fNet = fNet(net(input));
        bias = newBias(input, fNet);
        weights = newWeights(input, fNet);
    }

    private double[] newWeights(SetRow input, double fNet) {
        double[] newWeights = new double[weights.length];

        for (int i = 0; i < input.length(); i++) {

            newWeights[i] = weights[i] + learningRate * (input.getResult() - fNet) * input.getParameters(i);

        }
        return newWeights;
    }

    private double newBias(SetRow input, double fNet) {
        return bias - (learningRate * (input.getResult() - fNet));
    }

    private int fNet(double net) {
        return net >= 0 ? 1 : 0;
    }

    private double net(SetRow input) {

        double net = 0;

        for (int i = 0; i < input.length(); i++) {

            net += weights[i] * input.getParameters(i);

        }

        return net - bias;
    }
}
