
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions._

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Random, Success, Try}

case class Interaction(user: Int, artist: Int, count: Int)

object MusicRecommender extends App {

  val spark = SparkSession
    .builder()
    .appName("Music Reccommender")
    .config("spark.master", "local")
    .getOrCreate()
  import spark.implicits._

  Logger.getRootLogger.setLevel(Level.WARN)
  //spark.sparkContext.setLogLevel("ERROR")

  val base = "/home/tatjana/code/spark/datasets/audioscrobbler-2005/"
  val rawUserArtistData = spark.read.textFile(base + "user_artist_data.txt")
  val rawArtistData = spark.read.textFile(base + "artist_data.txt")
  val rawArtistAlias = spark.read.textFile(base + "artist_alias.txt")

  inspection(rawUserArtistData, rawArtistData, rawArtistAlias)
  model(rawUserArtistData, rawArtistData, rawArtistAlias)
  //evaluate(rawUserArtistData, rawArtistAlias)
  //recommend(rawUserArtistData, rawArtistData, rawArtistAlias)

  def inspection(rawUserArtistData: Dataset[String],
                  rawArtistData: Dataset[String],
                  rawArtistAlias: Dataset[String]): Unit = {

    rawUserArtistData.take(5).foreach(println)

    val userArtistDF: DataFrame = rawUserArtistData.map { line =>
      val Array(user, artist, _*) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")

    userArtistDF.agg(min($"user"), max($"user"), min($"artist"), max($"artist")).show()

    val artistByID = buildArtistByID(rawArtistData)
    val artistAlias = buildArtistAlias(rawArtistAlias)

    val (badID, goodID) = artistAlias.head
    artistByID.filter($"id" isin (badID, goodID)).show()
  }

  def model(rawUserArtistData: Dataset[String],
            rawArtistData: Dataset[String],
            rawArtistAlias: Dataset[String]): Unit = {

    val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))

    val data = buildCounts(rawUserArtistData, bArtistAlias).cache()

    val Array(train, test) = data.randomSplit(Array(0.8, 0.2))

    val als = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(10).
      setRegParam(0.01).
      setAlpha(1.0).
      setMaxIter(5).
      setUserCol("user").
      setItemCol("artist").
      setRatingCol("count").
      setPredictionCol("prediction")

    val model = als.fit(train)
    //model.save("music-model-2")
    //val model = ALSModel.load("music-model-2")

    train.unpersist()

    model.userFactors.select("features").show(truncate = false)

    // quick sanity check for a random user
//    val userID = 2093760
//    // 2030067, 1024631, 1059334
//
//    val existingArtistIDs = train.
//      filter($"user" === userID).
//      select("artist").as[Int].collect()
//
//    val artistByID = buildArtistByID(rawArtistData)
//
//    println("Artists user listened")
//    artistByID.filter($"id" isin (existingArtistIDs:_*)).show()
//
//    val users = train.filter($"user" === userID)
//    val userRecs: DataFrame = model.recommendForUserSubset(users, 5)
//    userRecs.show(truncate = false)
//
//    val recommendedations = userRecs.select("recommendations").as[Array[(Int, Float)]].collect()
//    val recommendedArtistIDs = recommendedations(0).map(_._1)
//
//    println("Artists recommended")
//    artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()

    val nRecommendations: Int = 10
    val testUsers = test.select($"user").distinct().limit(3)
    val recommendations = model.recommendForUserSubset(testUsers, nRecommendations)
    recommendations.show(10, truncate = false)
    println("Reccommendations size " + recommendations.count())
    val groundTruth = test
      .withColumn("actual", struct("count", "artist"))
      .groupBy($"user")
      .agg(slice(sort_array(collect_list($"actual").as("actual"), asc=false), 1, nRecommendations))
    val groundTruthArtists = groundTruth.map { case Row(user, actual: Array[(Int, Int)]) =>
      (user, actual.map(_._2)) }.toDF("user", "actual")
    groundTruth.show(10, truncate = false)
    groundTruthArtists.show(10, truncate = false)
    println("Ground truth size " + groundTruth.count())

    val relevantArtists = groundTruth.join(recommendations, usingColumn = "user")
    relevantArtists.show(10, truncate = false)
    println("Relevant artists " + relevantArtists.count())


//
//    val relevantDocuments = userMovies.join(userRecommended).map { case (user, (actual,
//    predictions)) =>
//      (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
//    }



    // https://stackoverflow.com/questions/37975715/rankingmetrics-in-spark-scala
//    val predictions = model.setPredictionCol("prediction").transform(test)
//    //predictions.take(10).foreach(println)
//
//    val testCountCol: Array[Int] = test.select("count").collect().map(_.getInt(0))
//    val testPredictionCol: Array[Float] = predictions.select("prediction").collect().map(_.getFloat(0))
//    val metrics: RDD[(Array[Float], Array[Int])] = spark.createDataset(List((testPredictionCol, testCountCol))).rdd
//    val rm = new RankingMetrics[Float](metrics)
//    println(s"Mean-average-precision = ${rm.meanAveragePrecision}")
  }

//  def truncateArray(toSize: Int) = udf { array: Seq[Row] =>
//    array.map(a => if (a.size < toSize) a else a.)
//  }

  def evaluate(rawUserArtistData: Dataset[String],
               rawArtistAlias: Dataset[String]): Unit = {

    val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))

    val allData = buildCounts(rawUserArtistData, bArtistAlias)
    val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    cvData.cache()

    val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
    val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)
  }

  def recommend(rawUserArtistData: Dataset[String],
                rawArtistData: Dataset[String],
                rawArtistAlias: Dataset[String]): Unit = {

    val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))
    val allData = buildCounts(rawUserArtistData, bArtistAlias).cache()
    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(10).setRegParam(1.0).setAlpha(40.0).setMaxIter(20).
      setUserCol("user").setItemCol("artist").
      setRatingCol("count").setPredictionCol("prediction").
      fit(allData)
    allData.unpersist()

    val userID = 2093760
    val topRecommendations = makeRecommendations(model, userID, 5)

    val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
    val artistByID = buildArtistByID(rawArtistData)
    artistByID.join(spark.createDataset(recommendedArtistIDs).toDF("id"), "id").
      select("name").show()

    model.userFactors.unpersist()
    model.itemFactors.unpersist()
  }

  def buildArtistByID(rawArtistData: Dataset[String]): DataFrame = {
    // span splits this sequence into a prefix/suffix pair according to a predicate
    rawArtistData.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      val maybePair = Try((id.toInt, name.trim)) match {
        case Success((id, name)) => Some((id, name))
        case Failure(_) => None
      }
      maybePair.filter { case (_, name) => !name.isEmpty }
    }.toDF("id", "name")
  }

  def buildArtistAlias(rawArtistAlias: Dataset[String]): Map[Int,Int] = {
    rawArtistAlias.flatMap { line =>
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.collect().toMap
  }

  def buildCounts(rawUserArtistData: Dataset[String],
                  bArtistAlias: Broadcast[Map[Int,Int]]): Dataset[Interaction] = {
    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Interaction(userID, finalArtistID, count)
    }
  }

  def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): DataFrame = {
    val toRecommend = model.itemFactors.
      select($"id".as("artist")).
      withColumn("user", lit(userID))
    model.transform(toRecommend).
      select("artist", "prediction").
      orderBy($"prediction".desc).
      limit(howMany)
  }

  def areaUnderCurve(positiveData: DataFrame,
                     bAllArtistIDs: Broadcast[Array[Int]],
                     predictFunction: (DataFrame => DataFrame)): Double = {

    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive".
    // Make predictions for each of them, including a numeric score
    val positivePredictions = predictFunction(positiveData.select("user", "artist")).
      withColumnRenamed("prediction", "positivePrediction")

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other artists, excluding those that are "positive" for the user.
    val negativeData = positiveData.select("user", "artist").as[(Int,Int)].
      groupByKey { case (user, _) => user }.
      flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs = bAllArtistIDs.value
        var i = 0
        // Make at most one pass over all artists to avoid an infinite loop.
        // Also stop when number of negative equals positive set size
        while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
          // Only add new distinct IDs
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // Return the set with user ID added back
        negative.map(artistID => (userID, artistID))
      }.toDF("user", "artist")

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction", "negativePrediction")

    // Join positive predictions to negative predictions by user, only.
    // This will result in a row for every possible pairing of positive and negative
    // predictions within each user.
    val joinedPredictions: DataFrame = positivePredictions.join(negativePredictions, "user").
      select("user", "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs per user
    val allCounts = joinedPredictions.
      groupBy("user").agg(count(lit("1")).as("total")).
      select("user", "total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy("user").agg(count("user").as("correct")).
      select("user", "correct")

    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").
      select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }

  def predictMostListened(train: DataFrame)(allData: DataFrame): DataFrame = {
    val listenCounts = train.groupBy("artist").
      agg(sum("count").as("prediction")).
      select("artist", "prediction")
    allData.
      join(listenCounts, Seq("artist"), "left_outer").
      select("user", "artist", "prediction")
  }

}
