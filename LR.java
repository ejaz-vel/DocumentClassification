import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LR {

	private Map<String, List<Double>> parameterWeights;
	private Map<String, List<Integer>> lastUpdated;
	private Double learningRate;
	private Double regularizationFactor;
	private Integer memSize;
	private Integer numOfIterations;
	private Integer trainingSetSize;
	private String testFile;
	private List<String> classLabels;

	public LR(String[] cmdArguments) {
		parameterWeights = new HashMap<>();
		lastUpdated = new HashMap<>();
		memSize = Integer.parseInt(cmdArguments[0]) / 100;
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
			parameterWeights.put(label, Arrays.asList(new Double[memSize]));
			lastUpdated.put(label, Arrays.asList(new Integer[memSize]));
		}
	}
	
	private void classify() {
		try {
			BufferedReader br = new BufferedReader(new FileReader(testFile));
			String testData = br.readLine();
			while(testData != null) {
				List<Integer> hashIndex = tokenizeStringtoHash(testData.split("\t")[1], "\\s+");
				//Find Probability for each label
				for(int i = 0; i < classLabels.size(); i++) {
					double probability = predict(classLabels.get(i), hashIndex);
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
					List<Integer> hashIndex = tokenizeStringtoHash(trainingData[1], "\\s+");
					for(String label : classLabels) {
						
						//Apply Lazy WeightDecay
						for(Integer hash: hashIndex) {
							if (parameterWeights.get(label).get(hash) == null) {
								parameterWeights.get(label).set(hash, 0.0);
							}
							if (lastUpdated.get(label).get(hash) == null) {
								lastUpdated.get(label).set(hash, 0);
							}
							double wordWeight = parameterWeights.get(label).get(hash);
							wordWeight *= Math.pow(1 - (2 * learningRate * regularizationFactor), 
												k - lastUpdated.get(label).get(hash));
							parameterWeights.get(label).set(hash, wordWeight);
						}
						
						//Predict
						double y = 0;
						if (trainingLabels.contains(label)) {
							y = 1;
						}
						double p = predict(label, hashIndex);
						
						//Apply gradient descent rule
						for(Integer hash: hashIndex) {
							double wordWeight = parameterWeights.get(label).get(hash);
							wordWeight += learningRate * (y - p);
							parameterWeights.get(label).set(hash, wordWeight);
							lastUpdated.get(label).set(hash, k);
						}
					}
				}
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// The below final weight decay does not affect accuracy by a great deal.
		// Furthermore, it's a dense update. Hence, ignoring it will save us a lot of time.
		for(String label: classLabels) {
			for (int hash = 0; hash < parameterWeights.get(label).size(); hash++) {
				Double wordWeight = parameterWeights.get(label).get(hash);
				if (wordWeight != null) {
					wordWeight *= Math.pow(1 - (2 * learningRate * regularizationFactor), 
									k - lastUpdated.get(label).get(hash));
					parameterWeights.get(label).set(hash, wordWeight);
				}
			}
		}
	}

	private List<String> tokenizeString(String string, String separator) {
		List<String> wordList = new ArrayList<>();
		String[] words = string.split(separator);
		for (String word: words) {
			wordList.add(word);
		}
		return wordList;
	}
	
	private List<Integer> tokenizeStringtoHash(String string, String separator) {
		List<Integer> hashIndex = new ArrayList<>();
		String[] words = string.split(separator);
		for (String word: words) {
			//Find Hash of the string
			int id = word.hashCode() % memSize;
			if (id<0) id += memSize;
			hashIndex.add(id);
		}
		return hashIndex;
	}

	private double predict(String label, List<Integer> hashIndex) {
		double dotProduct = 0;
		for (Integer hash: hashIndex) {
			if(parameterWeights.get(label).get(hash) != null) {
				dotProduct += parameterWeights.get(label).get(hash);
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
