from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

if __name__ == "__main__":
  sc = SparkContext(appName = "letter-recognition")

  # mapping alphabet to integers
  ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
  clean_value = lambda s: LabeledPoint(ALPHABET.index(s[0]), s.split(',')[1:])

  # read data and convert to labeled points
  raw_data = sc.textFile("hdfs:///data/letter-recognition.data")
  labeled_points = raw_data.map(clean_value)

  # separate into testing and training data
  (train, test) = labeled_points.randomSplit([0.7, 0.3])

  # build our model from the training data
  model = RandomForest.trainClassifier(train, numClasses=26, categoricalFeaturesInfo={},
                                       numTrees=50, featureSubsetStrategy="auto",
                                       impurity='gini', maxDepth=16, maxBins=32)


  # score the performance of our model on test data
  labels = test.map(lambda point: point.label)
  predicitons = model.predict(test.map(lambda point: point.features))
  labels_and_predictions = labels.zip(predicitons)

  total_correct = labels_and_predictions.filter(lambda x: x[0] == x[1])
  correct_percentage = total_correct.count() / float(labels_and_predictions.count())

  print ("PERCENTAGE CORRECTLY CLASSIFIED: {}".format(correct_percentage))

  sc.stop()