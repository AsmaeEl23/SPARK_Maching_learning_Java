package org.example;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KmeansApp {
    public static void main(String[] args) {
        SparkSession ss=SparkSession.builder().appName("Kmeans app").master("local[*]").getOrCreate();
        Dataset<Row> data=ss.read().option("inferSchema",true).option("header",true).csv("Mall_Customers.csv");
        VectorAssembler assembler=new VectorAssembler().setInputCols(
                new String[]{
                        "Age","Annual Income (k$)","Spending Score (1-100)"
                }
        ).setOutputCol("features");
        Dataset<Row> assembledDF=assembler.transform(data);
        //normalization
        MinMaxScaler scaler=new MinMaxScaler().setInputCol("features").setOutputCol("normalizedFeatures");
        Dataset<Row> normalizedDF =scaler.fit(assembledDF).transform(assembledDF);

        KMeans kMeans=new KMeans().setK(3)
                .setSeed(123)
                .setFeaturesCol("normalizedFeatures")
                .setPredictionCol("Cluster");
        KMeansModel model=kMeans.fit(normalizedDF);
        Dataset<Row> predictions=model.transform(normalizedDF);
        ClusteringEvaluator evaluator=new ClusteringEvaluator();
        double score=evaluator.evaluate(predictions);
        System.out.println("Score "+score);

    }
}
