import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LR {

	private static Map<String, Map<String,Double>> parameterWeights;
	private static Map<String, Map<String,Integer>> lastUpdated;
	private static int vocabSize;
	private static Double learningRate;
	private static Double regularizationFactor;
	private static int numOfIterations;
	private static int trainingSetSize;
	private static String testFile;
	private static List<String> classLabels;

	public LR(String[] cmdArguments) {
		parameterWeights = new HashMap<>();
		lastUpdated = new HashMap<>();
		vocabSize = Integer.parseInt(cmdArguments[0]);
		learningRate = Double.parseDouble(cmdArguments[1]);
		regularizationFactor = Double.parseDouble(cmdArguments[2]);
		numOfIterations = Integer.parseInt(cmdArguments[3]);
		trainingSetSize = Integer.parseInt(cmdArguments[4]);
		testFile = cmdArguments[5];
		classLabels = new ArrayList<>();
		classLabels.add("nl");
		classLabels.add("el");
		classLabels.add("ru");
		classLabels.add("sl");
		classLabels.add("pl");
		classLabels.add("ca");
		classLabels.add("fr");
		classLabels.add("tr");
		classLabels.add("hu");
		classLabels.add("de");
		classLabels.add("hr");
		classLabels.add("es");
		classLabels.add("ga");
		classLabels.add("pt");
		for (String label: classLabels) {
			parameterWeights.put(label, new HashMap<String,Double>());
			lastUpdated.put(label, new HashMap<String,Integer>());
		}
	}
	
	private void classify() {
		try {
			BufferedReader br = new BufferedReader(new FileReader(testFile));
			String testData = br.readLine();
			while(testData != null) {
				List<String> wordList = tokenizeString(testData.split("\t")[1], "\\s+");
				//Find Probability for each label
				for(int i = 0; i < classLabels.size(); i++) {
					double probability = predict(classLabels.get(i), wordList);
					System.out.print(classLabels.get(i) + "\t" + probability);
					if (i != (classLabels.size() - 1)) {
						System.out.print(",");
					}
				}
				System.out.println();
				testData = br.readLine();
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void trainSGD() {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		int k = 0;
		try {
			for(int i = 1; i <= numOfIterations; i++) {
				learningRate /= Math.pow(i, 2);
				for(int j = 0; j < trainingSetSize; j++) {
					k++;
					String[] trainingData = br.readLine().split("\t");
					List<String> trainingLabels = tokenizeString(trainingData[0], ",");;
					List<String> wordList = tokenizeString(trainingData[1], "\\s+");
					for(String label : classLabels) {
						
						//Apply Lazy WeightDecay
						for(String word: wordList) {
							if (parameterWeights.get(label).get(word) == null) {
								parameterWeights.get(label).put(word, 0.0);
							}
							if (lastUpdated.get(label).get(word) == null) {
								lastUpdated.get(label).put(word, 0);
							}
							double wordWeight = parameterWeights.get(label).get(word);
							wordWeight *= Math.pow(1 - (2 * learningRate * regularizationFactor), 
												k - lastUpdated.get(label).get(word));
							parameterWeights.get(label).put(word, wordWeight);
						}
						
						//Predict
						double y = 0;
						if (trainingLabels.contains(label)) {
							y = 1;
						}
						double p = predict(label, wordList);
						
						//Apply gradient descent rule
						for(String word: wordList) {
							double wordWeight = parameterWeights.get(label).get(word);
							wordWeight += learningRate * (y - p);
							parameterWeights.get(label).put(word, wordWeight);
							lastUpdated.get(label).put(word, k);
						}
					}
				}
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		for(String label: classLabels) {
			for (Map.Entry<String, Double> entry : parameterWeights.get(label).entrySet()) {
				double wordWeight = entry.getValue();
				wordWeight *= Math.pow(1 - (2 * learningRate * regularizationFactor), 
									k - lastUpdated.get(label).get(entry.getKey()));
				parameterWeights.get(label).put(entry.getKey(), wordWeight);
			}
		}
	}
	
	private List<String> tokenizeString(String string, String separator) {
		List<String> wordList = new ArrayList<>();
		String[] words = string.split(separator);
		for (String word: words) {
			word = word.replaceAll("\\W", "");
			if (word.length() > 0) {
				//Find Hash of the string
				//int id = word.hashCode() % vocabSize;
				//wordList.add(String.valueOf(id));
				wordList.add(word);
			}
		}
		return wordList;
	}

	private double predict(String label, List<String> wordList) {
		double dotProduct = 0;
		for (String word: wordList) {
			if(parameterWeights.get(label).containsKey(word)) {
				dotProduct += parameterWeights.get(label).get(word);
			};
		}
		return sigmoid(dotProduct);
	}

	private double sigmoid(double score) {
		double overflow = 20;
        if (score > overflow) {
        	score = overflow;
        } else if (score < -overflow) {
        	score = -overflow;
        }
        double exp = Math.exp(score);
        return exp / (1 + exp);
    }


	public static void main(String[] args) {
		LR sgd = new LR(args);
		sgd.trainSGD();
		sgd.classify();
	}

}
