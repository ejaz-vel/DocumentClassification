import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BatchLR {
	
	private Map<String, List<Double>> parameterWeights;
	private Map<String, List<Double>> gradientWeights;
	private Map<String, List<Integer>> lastUpdated;
	private Map<String, Double> squaredWeights;
	private Double learningRate;
	private Double regularizationFactor;
	private Integer memSize;
	//private Integer numOfIterations;
	private Integer trainingSetSize;
	private String testFile;
	private List<String> classLabels;

	public BatchLR(String[] cmdArguments) {
		parameterWeights = new HashMap<>();
		lastUpdated = new HashMap<>();
		gradientWeights = new HashMap<>();
		squaredWeights = new HashMap<>();
		memSize = Integer.parseInt(cmdArguments[0]);
		learningRate = Double.parseDouble(cmdArguments[1]);
		regularizationFactor = Double.parseDouble(cmdArguments[2]);
		//numOfIterations = Integer.parseInt(cmdArguments[3]);
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
			gradientWeights.put(label, Arrays.asList(new Double[memSize]));
			squaredWeights.put(label, 0.0);
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

	private void trainBatchGD() {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		int k = 0;
		Double minLoss = 0.0;
		try {
			for(int i = 1;; i++) {
				learningRate /= Math.pow(i, 2);
				initializeGradient();
				Double loss = 0.0;
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
							double wordWeight = gradientWeights.get(label).get(hash);
							wordWeight *= Math.pow(1 - (2 * learningRate * regularizationFactor), 
												k - lastUpdated.get(label).get(hash));
							gradientWeights.get(label).set(hash, wordWeight);
						}
						
						//Predict
						double y = 0;
						if (trainingLabels.contains(label)) {
							y = 1;
						}
						double p = predict(label, hashIndex);
						loss += (y * Math.log(p)) + ((1 - y) * Math.log(1-p)) - (regularizationFactor * squaredWeights.get(label));
						
						//Apply gradient descent rule
						for(Integer hash: hashIndex) {
							double wordWeight = gradientWeights.get(label).get(hash);
							wordWeight += learningRate * (y - p);
							gradientWeights.get(label).set(hash, wordWeight);
							lastUpdated.get(label).set(hash, k);
						}
					}
				}
				loss /= trainingSetSize;
				System.out.println("Value of Objective Function at Epoch " + i + ": " + String.valueOf(loss));
				if(Math.abs(loss - minLoss) < 0.0001) {
					//Convergence Achieved
					break;
				}
				minLoss = loss;
				updateWeights();
				computeSquaredWeights();
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void updateWeights() {
		for(String label: classLabels) {
			for (int hash = 0; hash < gradientWeights.get(label).size(); hash++) {
				Double weight = parameterWeights.get(label).get(hash);
				weight += (gradientWeights.get(label).get(hash) / trainingSetSize);
				parameterWeights.get(label).set(hash, weight);
			}
		}
	}

	private void initializeGradient() {
		for(String label: classLabels) {
			for (int hash = 0; hash < gradientWeights.get(label).size(); hash++) {
				gradientWeights.get(label).set(hash, 0.0);
			}
		}
	}

	private void computeSquaredWeights() {
		for (String label: classLabels) {
			double sum = 0;
			List<Double> weights = parameterWeights.get(label);
			for (Double weight : weights) {
				if(weight != null) {
					sum += Math.pow(weight, 2);
				}
			}
			squaredWeights.put(label, sum);
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
		BatchLR sgd = new BatchLR(args);
		sgd.trainBatchGD();
		sgd.classify();
	}

}
