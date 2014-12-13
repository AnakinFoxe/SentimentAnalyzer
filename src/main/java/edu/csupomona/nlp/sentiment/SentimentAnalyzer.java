package edu.csupomona.nlp.sentiment;

import edu.csupomona.nlp.util.MapUtil;

import java.io.*;
import java.util.*;

/**
 * Created by Xing HU on 10/26/14.
 */
public class SentimentAnalyzer {

    private HashMap<String, Integer> pos_;  // all positive n-grams
    private HashMap<String, Integer> neg_;  // all negative n-grams

    private long totalPos_;                 // total occurrence of positive n-grams
    private long totalNeg_;                 // total occurrence of negative n-grams

    private HashSet<String> features_;      // selected features (n-grams)

    private boolean needRetrain = false;            // need retrain the classifier or not
    private boolean needReselectFeature = false;    // need reselect features or not

    private String basePath = "data/";


    /**
     * Construct a SentimentAnalyzer instance
     */
    public SentimentAnalyzer() {
        pos_ = new HashMap<>();
        neg_ = new HashMap<>();

        totalPos_ = 0;
        totalNeg_ = 0;

        features_ = new HashSet<>();
    }

    public void setNeedRetrain(boolean needRetrain) {
        this.needRetrain = needRetrain;
    }

    public void setNeedReselectFeature(boolean needReselectFeature) {
        this.needReselectFeature = needReselectFeature;
    }

    public void setBasePath(String basePath) {
        this.basePath = basePath;
    }

    /**
     * "Safe" get value of the key from map
     * @param map               Input map
     * @param key               Input key
     * @return                  Value of the key. Returns 0 if the key does not exist
     */
    private int get(HashMap<String, Integer> map, String key) {
        return map.containsKey(key) ? map.get(key) : 0;
    }

    // TODO: use HashSet or List?
    // TODO: HashSet has slight higher accuracy on doc level than list

    /**
     * Tokenize the text and negate n-gram which has "not" in front of it.
     * @param text              Input text
     * @return                  Tokenized n-grams
     */
    private HashSet<String> negateSequence(String text) {
        boolean isNegation = false;
        String delims = "[,.?!;:]";
        HashSet<String> results = new HashSet<>();
        String[] words = text.toLowerCase().split(" ");
        String prev = null;
        String pprev = null;

        for (String word : words) {
            String stripped = word.replaceAll(delims, "").trim();
            if (stripped.length() == 0) // filter out null strings
                continue;
            String negated = isNegation ? "not_" + stripped : stripped;
            results.add(negated);

            if (prev != null) {
                String bigram = prev + " " + negated;
                results.add(bigram);

                if (pprev != null) {
                    String trigram = pprev + " " + bigram;
                    results.add(trigram);
                }
                pprev = prev;
            }
            prev = negated;

            if (word.equals("not") || word.equals("n't") || word.equals("no"))
                isNegation = !isNegation;

            if (!word.equals(word.replaceAll(delims, "")))
                isNegation = false;
        }

        return results;
    }

    private void pruneFeatures() {
        HashSet<String> posToBePruned = new HashSet<>();
        HashSet<String> negToBePruned = new HashSet<>();

        for (String key : pos_.keySet())
            if ((pos_.get(key) <= 1) && (!neg_.containsKey(key) || neg_.get(key) <= 1))
                posToBePruned.add(key);

        for (String key : posToBePruned)
            pos_.remove(key);

        for (String key : neg_.keySet())
            if ((neg_.get(key) <= 1) && (!pos_.containsKey(key) || pos_.get(key) <= 1))
                negToBePruned.add(key);

        for (String key : negToBePruned)
            neg_.remove(key);

    }

    private void loadTrainedData() throws IOException {
        // load pos_
        FileReader fr = new FileReader(basePath + "pos_.txt");
        try (BufferedReader br = new BufferedReader(fr)) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] items = line.split(",");
                pos_.put(items[0].trim(), Integer.valueOf(items[1]));
            }
        }

        // load neg_
        fr = new FileReader(basePath + "neg_.txt");
        try (BufferedReader br = new BufferedReader(fr)) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] items = line.split(",");
                neg_.put(items[0].trim(), Integer.valueOf(items[1]));
            }
        }

        // load total
        fr = new FileReader(basePath + "total.txt");
        try (BufferedReader br = new BufferedReader(fr)) {
            String line = br.readLine();
            totalPos_ = Integer.valueOf(line != null ? line : "0");
            line = br.readLine();
            totalNeg_ = Integer.valueOf(line != null ? line : "0");
        }
    }

    private void saveTrainedData() throws IOException {
        // save pos_
        FileWriter fw = new FileWriter(basePath + "pos_.txt");
        try (BufferedWriter bw = new BufferedWriter(fw)) {
            for (String word : pos_.keySet())
                bw.write(word + "," + pos_.get(word) + "\n");
        }

        // save neg_
        fw = new FileWriter(basePath + "neg_.txt");
        try (BufferedWriter bw = new BufferedWriter(fw)) {
            for (String word : neg_.keySet())
                bw.write(word + "," + neg_.get(word) + "\n");
        }

        // save total
        fw = new FileWriter(basePath + "total.txt");
        try (BufferedWriter bw = new BufferedWriter(fw)) {
            bw.write(totalPos_ + "\n");
            bw.write(totalNeg_ + "\n");
        }
    }

    public void train() throws IOException {
        int limit = 12500;
        int counter = 0;

        if (!needRetrain) {
            loadTrainedData();
            System.out.println("pos: " + totalPos_ + ", neg: " + totalNeg_);
            System.out.println("pos_size: " + pos_.size() + ", neg_size: " + neg_.size());
            return;
        }

        File[] posFiles = new File(basePath + "aclImdb/train/pos/").listFiles();
        File[] negFiles = new File(basePath + "aclImdb/train/neg/").listFiles();

        for (File file : posFiles) {
            FileReader fr = new FileReader(file);
            try (BufferedReader br = new BufferedReader(fr)) {
                String line;
                while ((line = br.readLine()) != null) {
                    // TODO: preprocess

                    // negate sequence
                    HashSet<String> words = negateSequence(line);
                    for (String word : words) {
                        MapUtil.updateHashMap(pos_, word);
                        MapUtil.updateHashMap(neg_, "not_" + word);
                    }
                }
            }

            if (++counter >= limit)
                break;

            if (counter % 1000 == 0)
                System.out.println("Processed positive files: " + counter);
        }

        counter = 0;
        for (File file : negFiles) {
            FileReader fr = new FileReader(file);
            try (BufferedReader br = new BufferedReader(fr)) {
                String line;
                while ((line = br.readLine()) != null) {
                    // TODO: preprocess

                    // negate sequence
                    HashSet<String> words = negateSequence(line);
                    for (String word : words) {
                        MapUtil.updateHashMap(neg_, word);
                        MapUtil.updateHashMap(pos_, "not_" + word);
                    }
                }
            }

            if (++counter >= limit)
                break;

            if (counter % 1000 == 0)
                System.out.println("Processed negative files: " + counter);
        }


        pruneFeatures();

        totalPos_ = MapUtil.sumHashMap(pos_);
        totalNeg_ = MapUtil.sumHashMap(neg_);
        System.out.println("pos: " + totalPos_ + ", neg: " + totalNeg_);

        saveTrainedData();
    }

    private String readText(String filePath) throws IOException {
        StringBuilder text = new StringBuilder();
        FileReader fr = new FileReader(filePath);
        try (BufferedReader br = new BufferedReader(fr)) {
            String line;
            while ((line = br.readLine()) != null) {
                text.append(line);
                text.append(" ");
            }
        }

        return text.toString();
    }

    /**
     * Classify the text into positive or negative class
     * @param text              Input text
     * @return                  True: positive, False: negative
     */
    public boolean classify(String text) {
        double probPos = 0.0;
        double probNeg = 0.0;
        for (String word : negateSequence(text)) {
            if (features_.contains(word)) {
                probPos += Math.log((get(pos_, word) + 1.0) / (2 * totalPos_));
                probNeg += Math.log((get(neg_, word) + 1.0) / (2 * totalNeg_));
            }
        }

        return probPos > probNeg;
    }

    public boolean classify2(String text) {
        double probPos = 0.0;
        double probNeg = 0.0;
        for (String word : negateSequence(text)) {
            if (pos_.containsKey(word) || neg_.containsKey(word)) {
                probPos += Math.log((get(pos_, word) + 1.0) / (2 * totalPos_));
                probNeg += Math.log((get(neg_, word) + 1.0) / (2 * totalNeg_));
            }
        }

        return probPos > probNeg;
    }

    private double computeMI(String word) {
        double t = totalPos_ + totalNeg_;
        double pos = get(pos_, word);
        double neg = get(neg_, word);
        double w = pos + neg;
        double i = 0.0;

        if (w != 0) {
            if (neg > 0) {
                i += (totalNeg_ - neg) / t * Math.log((totalNeg_ - neg) * t / (t - w) / totalNeg_);
                i += neg / t * Math.log(neg * t / w / totalNeg_);
            }
            if (pos > 0) {
                i += (totalPos_ - pos) / t * Math.log((totalPos_ - pos) * t / (t - w) / totalPos_);
                i += pos / t * Math.log(pos * t / w / totalPos_);
            }
        }

        return i;
    }

    private void loadFeatures() throws IOException {
        FileReader fr = new FileReader(basePath + "features_.txt");
        try (BufferedReader br = new BufferedReader(fr)) {
            String line;
            while ((line = br.readLine()) != null)
                features_.add(line.trim());
        }
    }

    private void saveFeatures() throws IOException {
        FileWriter fw = new FileWriter(basePath + "features_.txt");
        try (BufferedWriter bw = new BufferedWriter(fw)) {
            for (String feature: features_)
                bw.write(feature + "\n");
        }
    }

    public void selectFeatures() throws IOException {
        if (!needReselectFeature) {
            loadFeatures();
            return;
        }

        Map<String, Double> words = new TreeMap<>();

        // construct a map holding all words and their mutual information
        for (String word : pos_.keySet()) {
            if (!words.containsKey(word))
                words.put(word, computeMI(word));   // suppose to be -1 * MI, but changed for Map sorting issue
        }
        for (String word : neg_.keySet()) {
            if (!words.containsKey(word))
                words.put(word, computeMI(word));
        }

        System.out.println("Total number of features: " + words.size());

        // sort the map
        words = MapUtil.sortByValue(words);

        // prepare the test files
        File[] posFiles = new File(basePath + "aclImdb/test/pos/").listFiles();
        File[] negFiles = new File(basePath + "aclImdb/test/neg/").listFiles();
        List<String> posText = new ArrayList<>();
        List<String> negText = new ArrayList<>();
        for (File file : posFiles)
            posText.add(readText(file.getAbsolutePath()));
        for (File file : negFiles)
            negText.add(readText(file.getAbsolutePath()));

        // select features
        int bestK = 0;
        int limit = 500;
        int step = 500;
        int start = 20000;
        int end = start + 20000;
        double bestAccuracy = 0.0;
        for (String word : words.keySet()) {
            // directly add top words as features
            features_.add(word);

            // for rest words, test accuracy on testing data
            int featSize = features_.size();
            if ((featSize >= start) && ((featSize - start) % step == 0)) {
                // do the testing once reached the step point
                int correct = 0;
                for (String text : posText)
                    correct = classify(text) ? correct + 1 : correct;
                for (String text : negText)
                    correct = classify(text) ? correct : correct + 1;

                // record the point reaches highest accuracy
                double accuracy = (double) correct / (posText.size() + negText.size());
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestK = featSize;
                }

                System.out.println(featSize + ": " + accuracy);
            }

            // stop point
            if (featSize >= end)
                break;
        }

        // rebuild features_
        features_ = new HashSet<>();
        for (String word : words.keySet()) {
            features_.add(word);

            if (features_.size() >= bestK)
                break;
        }

        // TODO: it is strange that after feature selection,
        // we still use the old totalPos_ and totalNeg_.
        // Should we re-calculate them?

        saveFeatures();
    }

    public HashMap<String, List<Integer>> getFeatures() {
        HashMap<String, List<Integer>> features = new HashMap<>();

        for (String ngram : features_) {
            List<Integer> posNeg = new ArrayList<>();

            // TODO: is it possible the key does not exist?
            if (pos_.containsKey(ngram))
                posNeg.add(pos_.get(ngram));    // pos
            else
                posNeg.add(0);

            if (neg_.containsKey(ngram))
                posNeg.add(neg_.get(ngram));    // neg
            else
                posNeg.add(0);

            // key : pos + neg mapping
            features.put(ngram, posNeg);
        }

        return features;
    }

    public List<Long> getTotals() {
        List<Long> posNegTotal = new ArrayList<>();

        // this is a modification on original algorithm
        // which does not remove features that were not going to be used
        Long totalPos = 0L;
        Long totalNeg = 0L;
        for (String ngram : features_) {
            if (pos_.containsKey(ngram))
                totalPos += pos_.get(ngram);

            if (neg_.containsKey(ngram))
                totalNeg += neg_.get(ngram);
        }

        posNegTotal.add(totalPos); // pos
        posNegTotal.add(totalNeg); // neg

        return posNegTotal;
    }

    public void simpleTrain(HashMap<String, Integer> pos,
                            HashMap<String, Integer> neg,
                            Long totalPos, Long totalNeg,
                            HashSet<String> features) {
        pos_ = pos;
        neg_ = neg;
        totalPos_ = totalPos;
        totalNeg_ = totalNeg;
        features_ = features;
    }

    public void testBoPang() throws IOException {
        File[] posFiles = new File(basePath + "txt_sentoken/pos/").listFiles();
        File[] negFiles = new File(basePath + "txt_sentoken/neg/").listFiles();
        List<String> posText = new ArrayList<>();
        List<String> negText = new ArrayList<>();
        for (File file : posFiles)
            posText.add(readText(file.getAbsolutePath()));
        for (File file : negFiles)
            negText.add(readText(file.getAbsolutePath()));

        int correct = 0;
        for (String text : posText)
            correct = classify2(text) ? correct + 1 : correct;
        for (String text : negText)
            correct = classify2(text) ? correct : correct + 1;

        // record the point reaches highest accuracy
        double accuracy = (double) correct / (posText.size() + negText.size());

        System.out.println(accuracy);
    }


    public static void main(String[] args) throws IOException {
        SentimentAnalyzer sa = new SentimentAnalyzer();

        sa.train();

        sa.selectFeatures();

        sa.testBoPang();

//        Map<String, Double> test = new TreeMap<>();
//        test.put("1", 1.0);
//        test.put("3", 3.0);
//        test.put("2", 2.0);
//
//        test = MapUtil.sortByValue(test);
//        for (String key : test.keySet())
//            System.out.println(key);
    }
}