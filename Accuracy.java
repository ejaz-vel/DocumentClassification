import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Accuracy {
	public static void main(String args[]) throws IOException {
		String predictionFile = args[0];
		String testFile = args[1];
		BufferedReader br1 = new BufferedReader(new FileReader(predictionFile));
		BufferedReader br2 = new BufferedReader(new FileReader(testFile));
		String prediction = br1.readLine();
		String test = br2.readLine();
		int total = 0;
		int truePositvies = 0;
		int trueNegatives = 0;
		while(prediction != null && test != null) {
			List<String> correctLabels = Arrays.asList(test.split("\t")[0].split(","));
			List<String> predictionProb = Arrays.asList(prediction.split(","));
			for (String prob: predictionProb) {
				String[] info = prob.split("\t");
				String label = info[0];
				Double probability = Double.parseDouble(info[1]);
				if(probability >= 0.5) {
					if(correctLabels.contains(label)) {
						truePositvies++;
					}
				} else {
					if(!correctLabels.contains(label)) {
						trueNegatives++;
					}
				}
			}
			total += 14;
			prediction = br1.readLine();
			test = br2.readLine();
		}
		
		br1.close();
		br2.close();
		
		System.out.println("True Positives: " + String.valueOf(truePositvies));
		System.out.println("True Negatives: " + String.valueOf(trueNegatives));
		System.out.println("Total Labels: " + String.valueOf(total));
		System.out.println("Accuracy: " + String.valueOf((truePositvies + trueNegatives) / (total + 0.0)));
	}
}
