title: Machine Learning at Scale
author:
  name: Nathan Epstein
  twitter: epstein_n
  url: http://nepste.in
  email: _@nepste.in

--

# Machine Learning at Scale

--

### Goals

1) Understand (some of) the ecosystem of tools for machine learning with big data.

2) Be able to apply predictive models to large data sets on a remote server cluster.

--

### Tools
- AWS
- Spark / PySpark
- MLlib
- Hadoop Distributed File System
- flintrock (optional)
--

### AWS

 Amazon Web Services - A suite of cloud computing services that make up an on-demand computing platform.

--

### AWS - Elasic Compute Cloud (EC2)

Rentable virtual computers on which individuals can run their own applications.

--

### AWS - Elastic MapReduce (EMR)

A managed Hadoop framework to distribute and process data across dynamically scalable Amazon EC2 instances.

--

### AWS - Simple Storage Service (S3)

Durable and highly-scalable cloud storage.

--

### Spark

- An open source cluster computing framework.
- Compatible with Python (as PySpark).
- Also available for Java, Scala, and R.

--

### MLlib

- Machine learning library built on top of on Spark.
- Compatible with Python / PySpark (as well as Java, Scala, and SparkR).
- Python APIs will be comfortable for users of scikit-learn.

--

### Hadoop Distributed File System (HDFS)

- Java-based file system for scalable and reliable data storage (which Spark uses).
- Highly fault tolerant; architected to expect failure of individual nodes.
- High bandwidth for processing large files quickly.

--

### flintrock

- A command-line tool for launching Spark clusters on AWS.
- One of several such tools; clusters can also be managed from an AWS CLI or web console.

--

### Our Big Data ML Stack

<img height=350 width=450 src="https://raw.githubusercontent.com/NathanEpstein/pydata-berlin/master/images/stack.png">


--

### Spark Crash Course

- Resilient Distributed Dataset (RDD)
- Partitioning
- "Array" Methods

--

### Resilient Distributed Dataset (RDD)

- Central structure in Spark. A read-only multiset of data distributed over a cluster of machines.
- Allows for common array methods to be performed in parallel but abstracts away this complexity.

--

### Partitioning

``` Python
from pyspark import SparkContext as sc

# create an RDD from a list
my_rdd = sc.parallelize([1, 2, 3, 4, 5])

# or create one by reading in data
another_rdd = sc.textFile("/path/to/some/data")

# "gather" the elements of the RDD
my_rdd.collect()
# output: [1, 2, 3, 4, 5]

```

--

<img height=350 width=600 src="https://raw.githubusercontent.com/NathanEpstein/pydata-berlin/master/images/partition.png">

--

### "Array" Methods

- map
- reduce
- filter

--

### Map

<img height=350 width=600 src="https://raw.githubusercontent.com/NathanEpstein/pydata-berlin/master/images/map.png">

--

### Reduce

<img height=350 width=600 src="https://raw.githubusercontent.com/NathanEpstein/pydata-berlin/master/images/reduce.png">


--

### Filter

<img height=350 width=600 src="https://raw.githubusercontent.com/NathanEpstein/pydata-berlin/master/images/filter.png">


--

### Chaining Methods

Chaining works basically how you would expect.

``` Python
even = lambda x: x % 2 == 0
double = lambda x: x * 2
add = lambda x, y: x + y

my_rdd.filter(even).map(double).reduce(add) #output: 12

"""
[1, 2] => [2] => [4] => 4
[3]    => []  => []  => 0 => 4 + 0 + 8 => 12
[4, 5] => [4] => [8] => 8
"""
```
--

Good news! No more drawing!

--

### MLlib Crash Course

- Like scikit-learn, MLlib supports common algorithms in supervised and unsupervised learning.
- Models are given training data in a specific format and can then make predictions on new points.
- It does other stuff like summary statistics / hypothesis testing / etc. There are a lot of features we can't cover.

--

### Supervised Example (Linear Regression)

```python

data = sc.textFile("path/to/some.data")
parsedData = data.map(parsePoint)

# Build the model
model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)

# Make a prediction for a new point
print(model.predict(p))

```
--

### Unsupervised Example (K-means Clustering)

``` python
data = sc.textFile("path/to/some.data")
parsedData = data.map(parsePoint)

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10,
        runs=10, initializationMode="random")

# Predict the cluster for a new point
print(clusters.predict(p))

```
--

### Sample Problem

Analyze a classification algorithm to a large data set using Spark / MLlib running on an AWS cluster.

--
### Our Data

```
T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8
I,5,12,3,7,2,10,5,5,4,13,3,9,2,8,4,10
D,4,11,6,8,6,10,6,2,6,10,3,7,3,7,3,9
N,7,11,6,6,3,5,9,4,6,4,4,10,6,10,2,8
G,2,1,3,1,1,8,6,6,6,6,5,9,1,7,5,10
S,4,11,5,8,3,8,8,6,9,5,6,6,0,8,9,7
B,4,2,5,4,4,8,7,6,6,7,6,6,2,8,7,10
A,1,1,3,2,1,8,2,2,2,8,2,8,1,6,2,7
J,2,2,4,4,2,10,6,2,6,12,4,8,1,6,1,7
M,11,15,13,9,7,13,2,6,2,12,1,9,8,1,1,8
X,3,9,5,7,4,8,7,3,8,5,6,8,2,8,6,7
O,6,13,4,7,4,6,7,6,3,10,7,9,5,9,5,8
.          .          .          .
.          .          .          .
.          .          .          .
.          .          .          .
```

--

### Our Code (dependencies & variables)

``` python
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

if __name__ == "__main__":
  sc = SparkContext(appName = "letter-recognition")

  # mapping alphabet to integers
  ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
  clean_value = lambda row: LabeledPoint(ALPHABET.index(row[0]), row.split(',')[1:])
```
--

### Our Code (data, holdouts, & training)
```python
  # read data and convert to labeled points
  raw_data = sc.textFile("hdfs:///data/letter-recognition.data")
  labeled_points = raw_data.map(clean_value)

  # separate into testing and training data
  (train, test) = labeled_points.randomSplit([0.7, 0.3])

  # build our model from the training data
  model = RandomForest.trainClassifier(train, numClasses=26, categoricalFeaturesInfo={},
                                       numTrees=50, featureSubsetStrategy="auto",
                                       impurity='gini', maxDepth=16, maxBins=32)
```

--

### Our Code (scoring & output)

```python
  # score the performance of our model on test data
  labels = test.map(lambda point: point.label)
  predicitons = model.predict(test.map(lambda point: point.features))
  labels_and_predictions = labels.zip(predicitons)

  total_correct = labels_and_predictions.filter(lambda x: x[0] == x[1])
  correct_percentage = total_correct.count() / float(labels_and_predictions.count())

  print ("PERCENTAGE CORRECTLY CLASSIFIED: {}".format(correct_percentage))
  """
  You may want to do something else here.
  For example, write the model to a file which can be exported to S3.
  """

  sc.stop()
```
--

### Putting It Together (Procedure)

* Setup flintrock
* Launch a cluster
* Setup the cluster
* Get the files on the cluster
* Put files on the HDFS
* Run the code

--

### Setup flintrock

- install flintrock following instructions on Github
- `flintrock configure`
- in the YAML file
  - set AWS credentials
  - adjust AWS instance as desired (type, slave node count, region, etc.)
  - set `install-hdfs` to `True`

--

### Launch a cluster

- `flintrock launch my_cluster`
- `flintrock login my_cluster`

--

### Setup the cluster

The machines in our cluster require some setup (i.e. we need to install dependencies)

```
sudo yum install -y gcc
sudo pip install numpy
```
--

### Get the files on the cluster

- S3 is ideal for hosting large files, copy from there. AWS CLI syntax is:
```
aws s3 cp s3://bucket/file ./local/path
```

- Grab our data and code:
```
aws s3 cp s3://python-nepstein/letter-recognition.data ./letter-recognition.data
aws s3 cp s3://python-nepstein/classify-letters.py ./classify-letters.py
```

--

### Put files on the HDFS
```
hadoop/bin/hadoop fs -put letter-recognition.data /
hadoop/bin/hadoop fs -put classify-letters.py /
```
--

### Run the code

Lastly...

```
spark/bin/spark-submit hdfs:///classify-letters.py

```

... and done!