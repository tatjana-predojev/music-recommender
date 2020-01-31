
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Random, Success, Try}

case class Interaction(user: Int, artist: Int, count: Int)
case class RankingHelper(user: Int, actual: Array[(Int, Int)], recommendations: Array[(Int, Float)])
case class RankingLabels(actual: Array[Int], recommendations: Array[Int])

object MusicRecommender extends App {

  val spark = SparkSession
    .builder()
    .appName("Music Recommender")
    .config("spark.master", "local")
    .getOrCreate()
  import spark.implicits._

  Logger.getRootLogger.setLevel(Level.WARN)
  //spark.sparkContext.setLogLevel("ERROR")

  val base = "/home/tatjana/code/spark/datasets/audioscrobbler-2005/"
  val rawUserArtistData = spark.read.textFile(base + "user_artist_data.txt")
  val rawArtistData = spark.read.textFile(base + "artist_data.txt")
  val rawArtistAlias = spark.read.textFile(base + "artist_alias.txt")

  // basic data checks
  inspection(rawUserArtistData, rawArtistData, rawArtistAlias)

  val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))

  val data = buildCounts(rawUserArtistData, bArtistAlias)
    .filter($"artist" =!= 1034635).cache() // remove [unknown] artist with id 1034635

  val Array(train, test) = data.randomSplit(Array(0.9, 0.1))

  // Alternative Least Squares, spark's collaborative filtering algorithm
  val als = new ALS()
    .setSeed(Random.nextLong())
    .setImplicitPrefs(true)
    .setRank(10)
    .setRegParam(0.01)
    .setAlpha(1.0)
    .setMaxIter(5)
    .setUserCol("user")
    .setItemCol("artist")
    .setRatingCol("count")
    .setPredictionCol("prediction")

  val model = als.fit(train)
  //model.save("music-model-2")
  //val model = ALSModel.load("music-model-2")

  val artistByID = buildArtistByID(rawArtistData)
  // quick sanity check for a random user
  quickRecommendationCheck(2093760)
  quickRecommendationCheck(2030067)
  quickRecommendationCheck(1024631)
  quickRecommendationCheck(1059334)

  val nRecommendations: Int = 10
  // recommend for all test users to calculate precision
  val testUsers = test.select($"user").distinct().limit(10)
  model.setColdStartStrategy("drop")
  val recommendations = model.recommendForUserSubset(testUsers, nRecommendations)
  recommendations.show(10, truncate = false)
  println("Reccommendations size " + recommendations.count())
  val groundTruth = test
    .withColumn("actual", struct("count", "artist"))
    .groupBy($"user")
    .agg(slice(sort_array(collect_list($"actual"), asc=false), 1, nRecommendations).as("actual"))
  groundTruth.show(10, truncate = false)
  println("Ground truth size " + groundTruth.count())

  val bothRecsAndTruth = groundTruth.join(recommendations, usingColumn = "user")
  bothRecsAndTruth.show(truncate = false)
  val labels = bothRecsAndTruth.as[RankingHelper].map(r =>
    RankingLabels(r.recommendations.map(_._1), r.actual.map(_._2)))
    .as[(Array[Int], Array[Int])]
  labels.show(truncate = false)
  val metrics = new RankingMetrics(labels.rdd)
  println("Precision " + metrics.precisionAt(5))

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

  def quickRecommendationCheck(userID: Int): Unit = {

    val existingArtistIDs = train.
      filter($"user" === userID).
      select("artist").as[Int].collect()

    println(s"Artists user ${userID} listened")
    artistByID.filter($"id" isin (existingArtistIDs:_*)).show(truncate = false)

    val user = train.filter($"user" === userID)
    val userRecs: DataFrame = model.recommendForUserSubset(user, 5)

    val recommendedArtistIDs = userRecs.select("recommendations").as[Array[(Int, Float)]]
      .flatMap(ar => ar.map(_._1)).collect()

    println(s"Artists recommended to user ${userID}")
    artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show(truncate = false)

    //artistByID.join(spark.createDataset(recommendedArtistIDs).toDF("id"), "id").
    //  select("name").show()
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

  def predictMostListened(train: DataFrame)(allData: DataFrame): DataFrame = {
    val listenCounts = train.groupBy("artist").
      agg(sum("count").as("prediction")).
      select("artist", "prediction")
    allData.
      join(listenCounts, Seq("artist"), "left_outer").
      select("user", "artist", "prediction")
  }

}
