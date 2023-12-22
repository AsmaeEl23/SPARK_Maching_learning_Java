package org.example;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Linearregression {
    public static void main(String[] args) {
        SparkSession ss=SparkSession.builder().appName("tp spark ml").master("local[*]").getOrCreate();
        Dataset<Row> dataset =ss.read().option("inferSchema",true).option("header",true).csv("advertising.csv");
        //ramcer les colons dans un vector
        VectorAssembler assembler=new VectorAssembler().setInputCols(
                new String[]{"TV","Radio","Newspaper"}
        ).setOutputCol("features");
        Dataset<Row> assembleDS=assembler.transform(dataset);
        Dataset<Row>  splits[]=assembleDS.randomSplit(new double[]{0.8,0.2},123);
        Dataset<Row> train=splits[0];
        Dataset<Row> test=splits[1];

        //ml model
        LinearRegression lr=new LinearRegression().setLabelCol("Sales").setFeaturesCol("features");
        LinearRegressionModel model=lr.fit(train);
        Dataset<Row> predictions=model.transform(test);
        predictions.show();
        System.out.println("intercept : "+model.intercept());
        System.out.println("coefficients : "+model.coefficients());

    }
}