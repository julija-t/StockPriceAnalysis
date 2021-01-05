package com.github.julijat.spark.stocks

import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

/** Computes the average daily return of every stock for every date.
 * Saves the results to the file as Parquet, CSV and to Sqlite database
 * Computes which stock was traded most frequently - as measured by closing price * volume - on average
 * Computes which stock was the most volatile as measured by annualized standard deviation of daily returns
 * Builds ML Linear regression model trying to predict the next day's price
 * Builds ML Decision Tree and Random Forest models trying to predict if next day's price went UP/DOWN or remained UNCHANGED
 */
object StockPriceAnalysis extends App{
  // Set the log level to only print errors
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

  val sourceData = if(args.nonEmpty) args(0) else "src/main/resources/stock_prices.csv"

  // Create a SparkSession
  val spark: SparkSession = SparkSession.builder()
    .appName("FullNews")
    .master("local[*]")
    .getOrCreate()
  println(s"Session started on Spark version ${spark.version}")

  // Read data from csv file to spark DataFrame with supplied schema
  val df = spark
    .read
    .format("csv")
    .option("inferSchema","true")
    .option("header", "true")
    .load(sourceData)
    .withColumn("date", col("date").cast("date"))
  df.printSchema()
  println(Console.BOLD + "\nSource Data" + Console.RESET)
  df.show(10, truncate = false)

  // Define window
  val windowSpec = Window
    .partitionBy("ticker")
    .orderBy("date")

  // Calculate daily returns using lag window function
  // lag returns the value in e / columnName column that is offset records before the current record.
  // lag returns null value if the number of records in a window partition is less than offset or defaultValue
  val dailyReturnsDf = df.withColumn("return", (col("close") - lag("close", 1).over(windowSpec)) / lag("close", 1).over(windowSpec))

  // Perform aggregations
  val avgReturnDf = dailyReturnsDf
  .groupBy("date").agg(mean("return").as("average_return")).sort("date")

  println(Console.BOLD + "\nAverage daily return of stocks by date" + Console.RESET)
  avgReturnDf.show(false)

  //Save df to parquet file
  avgReturnDf
    .coalesce(1)
    .write
    .format("parquet")
    .option("header", true)
    .mode("overwrite")
    .save("./src/main/resources/stock_average_return.parquet")

  // Save df to csv file
  avgReturnDf
    .coalesce(1)
    .write
    .format("csv")
    .option("header", true)
    .mode("overwrite")
    .save("./src/main/resources/stock_average_return.csv")

  // Save df to a Sqlite database
  import java.sql.DriverManager
  val driver = "org.sqlite.JDBC"
  val url = "jdbc:sqlite:/C:/sqlite/db/stock_analysis.db"
  val connection = DriverManager.getConnection(url)
  connection.isClosed()
  val props = new Properties()
  props.setProperty("driver", driver)
  avgReturnDf
    .withColumn("date", col("date").cast("string"))
    .coalesce(1)
    .write.mode("overwrite")
    .option("createTableColumnTypes", "date DATE, average_return DOUBLE")
    .jdbc(url, "AverageReturns", props)
  connection.close()

  println(Console.BOLD + "\nStocks by trading frequency" + Console.RESET)
  // Calculate which stock was traded most frequently - as measured by closing price * volume - on average
  df.withColumn("frequency", col("close") * col("volume"))
    .groupBy("ticker")
    .agg(
      mean("frequency").as("trading_frequency")
    )
    .orderBy(desc("trading_frequency"))
    .show(false)

  // The NYSE and NASDAQ average about 253 trading days a year
  val tradingDays: Int = 253

  // Calculate which stock was the most volatile as measured by annualized standard deviation of daily returns
  val volatilityDf = dailyReturnsDf
    .groupBy("ticker")
    .agg(
      stddev("return").as("daily_stddev")
    )
    .withColumn("variance", pow(col("daily_stddev"), 2))
    .withColumn("annualized_variance", col("variance") * tradingDays)
    .withColumn("annualized_stddev", sqrt(col("annualized_variance")))
    .orderBy(desc("annualized_stddev"))

  println(Console.BOLD + "\nStocks by volatility" + Console.RESET)
  volatilityDf.show(false)



  // ML
  // Linear Regression
  // Add a new column with previous day's close price
  val modifiedDf = df
    .withColumn("previous_close", lag(col("close"), offset = 1).over(windowSpec))
    .withColumn("week_day", date_format(col("date"), "E"))
    .na.drop

  // Index labels
  val indexer = new StringIndexer()
    .setInputCols(Array("ticker", "week_day"))
    .setOutputCols(Array("ticker_index", "week_day_index"))
    .fit(modifiedDf)

  val transformedDf = indexer.transform(modifiedDf)

  // Encode features
  val oneHotEncoder = new OneHotEncoder()
    .setInputCols(Array("ticker_index", "week_day_index"))
    .setOutputCols(Array("ticker_vector", "week_day_vector"))

  val encodedDf = oneHotEncoder.fit(transformedDf).transform(transformedDf)

  // Create a vector with all the necessary features
  val va = new VectorAssembler()
//    .setInputCols(Array("previous_close", "ticker_vector", "week_day_vector"))
    .setInputCols(Array("previous_close", "ticker_vector"))
    .setOutputCol("features")

  // Apply VectorAssembler
  val featuredDf = va.transform(encodedDf)

  // Scale features using feature Normalizer
  val normalizer = new Normalizer()
    .setInputCol("features")
    .setOutputCol("norm_features")
    .setP(2.0) //Function setP(2.0) has ensured that Euclidean norm will be conducted on features dataset.

  // Final Df prepared for ML
  val preparedDf = normalizer.transform(featuredDf)
  println(Console.BOLD + "\nDF prepared for ML" + Console.RESET)
  preparedDf.show(10, truncate = false)

  // Split the data into training and test sets (20% held out for testing)
  val Array(train, test) = preparedDf.randomSplit(Array(0.8, 0.2),seed = 2020)

  // Train a LinearRegression model
  val lr = new LinearRegression()
    .setLabelCol("close")
    .setFeaturesCol("norm_features")
//    .setMaxIter(50)
//    .setElasticNetParam(1.0)

  // Train model
  val lrModel = lr.fit(train)
  // Make predictions
  val lrPredictions = lrModel.transform(test)
  println(Console.BOLD + "\nLinear Regression model:\n" + Console.RESET)
  // Select example rows to display.
  lrPredictions.select("date", "ticker", "close", "prediction").sort(expr("RAND(42)")).show(false)

  // Print the coefficients and intercept for linear regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // Summarize the model over the training set and print out some metrics
  val trainingSummary = lrModel.summary
  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
  trainingSummary.residuals.show()
  //Root Mean Square Error
  println(s"Root Mean Square Error (RMSE): ${trainingSummary.rootMeanSquaredError}")
  //Coefficient of determination
  println(s"r2: ${trainingSummary.r2}")
  // Mean absolute error
  println(s"MAE = ${trainingSummary.meanAbsoluteError}")
  // Explained variance
  println(s"Explained variance = ${trainingSummary.explainedVariance}")



  // Decision Tree
  // Add a new column with label for close price UP/DOWN/UNCHANGED
  val classificationDf = preparedDf
    .withColumn("return_label",
      when(col("close").equalTo(col("previous_close")), "UNCHANGED")
        .otherwise(when(col("close") > col("previous_close"), "UP")
        .otherwise("DOWN"))
    )

  // Index labels, adding metadata to the label column
  // Fit on whole dataset to include all labels in index
  val labelIndexer = new StringIndexer()
    .setInputCol("return_label")
    .setOutputCol("label")
    .fit(classificationDf)

  // Split the data into training and test sets (20% held out for testing)
  val Array(trainingData, testData) = classificationDf.randomSplit(Array(0.8, 0.2),seed = 2020)

  // Train a DecisionTree model
  val dt = new DecisionTreeClassifier()
    .setFeaturesCol("norm_features")
    .setLabelCol("label")

  // Convert indexed labels back to original labels
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predicted_return_label")
    .setLabels(labelIndexer.labelsArray(0))

  // Chain indexers and tree in a Pipeline
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, dt, labelConverter))

  // Set parameters for Cross Validation
  val params = new ParamGridBuilder()
    .addGrid(dt.maxDepth, Array(1, 2, 3, 4, 5))
    .addGrid(dt.impurity, Array("entropy", "gini")) //how many columns to use for making decision
    .build()

  // Obtain evaluator. Evaluator will check label and prediction and see percentage of accurate answers
  val evaluator = new MulticlassClassificationEvaluator()
    .setMetricName("accuracy")

  // Create 2-fold CrossValidator
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(params)
    .setNumFolds(2)  // Use 3+ in practice. How many folds = how many loops
    .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

  // Train model. This also runs the indexers
  val dtModel = cv.fit(trainingData)

  println(Console.BOLD + "\n\nDecision Tree model:\n" + Console.RESET)
  // Select the best model
  val bestModel = dtModel.bestModel
  // Make predictions
  val dtPredictions = bestModel.transform(testData)
  // Select example rows to display.
  dtPredictions.select("features", "return_label", "predicted_return_label").sort(expr("RAND(42)")).show(false)
  // Compute test error
  val accuracy = evaluator.evaluate(dtPredictions)
  println(s"Test Error = ${1.0 - accuracy}")



  // Random Forest
  // Train a RandomForest model
  val rf = new RandomForestClassifier()
//    .setFeaturesCol("indexed_features")
    .setFeaturesCol("norm_features")
    .setLabelCol("label")

  // Chain indexer, rf and labelConverter in a Pipeline
  val rfPipeline = new Pipeline()
    .setStages(Array(labelIndexer, rf, labelConverter))

  // Set parameters for Cross Validation
  val rfParams = new ParamGridBuilder()
    .addGrid(rf.numTrees, Array(2, 5, 10, 20))
    .addGrid(rf.impurity, Array("entropy", "gini")) //how many columns to use for making decision
    .addGrid(rf.bootstrap, Array(true, false))
    .build()

  // Create 2-fold CrossValidator
  val rcv = new CrossValidator()
    .setEstimator(rfPipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(rfParams)
    .setNumFolds(2)  // Use 3+ in practice. How many folds = how many loops
    .setParallelism(8)

  // Train model. This also runs the indexer.
  val rfModel = rcv.fit(trainingData)

  println(Console.BOLD + "\n\nRandom Forest model:\n" + Console.RESET)
  // Select the best model
  val bestRfModel = rfModel.bestModel
  // Make predictions
  val rfPredictions = bestRfModel.transform(testData)
  // Select example rows to display.
  rfPredictions.select("features", "return_label", "predicted_return_label").sort(expr("RAND(42)")).show(false)
  // Compute test error
  val accuracyRf = evaluator.evaluate(rfPredictions)
  println(s"Test Error = ${1.0 - accuracyRf}")


}
