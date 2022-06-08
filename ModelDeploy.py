
import pyspark
import numpy as np
from flask import Flask, request, jsonify, render_template 

#Create the flask app

app = Flask(__name__)

#loading the model

sc = pyspark.SparkContext('local','SimpleApp')

spark = pyspark.sql.SparkSession.builder \
    .appName("Deploy") \
    .getOrCreate()

from pyspark.ml.classification import LogisticRegressionModel
import os

model_path = os.path.join('..', 'models', 'logModel')
model = LogisticRegressionModel.load(model_path)

@app.route("/")

def Home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    
    int_features = [x for x in request.form.values()]
    columns = ['state',
     'number_customer_service',
     'account_length',
     'international_plan',
     'voice_mail_plan',
     'number_vmail_messages',
     'total_day_minutes',
     'total_day_calls',
     'total_eve_minutes',
     'total_eve_calls',
     'total_night_minutes',
     'total_night_calls',
     'total_intl_minutes',
     'total_intl_calls',
     'number_customer_service_calls']
    
    df = spark.createDataFrame(int_features).toDF(*columns)
       
    df = df.withColumn(colName = "international_plan",col = when(df.international_plan == "no",0)\
                                  .otherwise(1))\
               .withColumn(colName = 'voice_mail_plan', col =  when(df.voice_mail_plan == "no", 0)\
                           .otherwise(1))\
               .withColumn(colName = "churn", col = when(df.churn == "no",0)\
                          .otherwise(1))
    
    from pyspark.ml.feature import StringIndexer
    
    stringIndexer = StringIndexer()\
                    .setInputCol('state')\
                    .setOutputCol('state_index')
    
    stateIndex = stringIndexer.fit(df)
    stateIndex = stateIndex.transform(df)
    
    from pyspark.ml.feature import OneHotEncoderEstimator

    encoder = OneHotEncoderEstimator()\
            .setInputCols(['state_index'])\
            .setOutputCols(['state_encoded'])

    state_encoded = encoder.fit(stateIndex)
    state_encoded = state_encoded.transform(stateIndex)
    
    assembler = VectorAssembler().setInputCols([
     'account_length',
     'international_plan',
     'voice_mail_plan',
     'number_vmail_messages',
     'total_day_minutes',
     'total_day_calls',
     'total_eve_minutes',
     'total_eve_calls',
     'total_night_minutes',
     'total_night_calls',
     'total_intl_minutes',
     'total_intl_calls',
     'number_customer_service_calls',
     'state_encoded']).setOutputCol("vectorized_features")
    
    assembled_df = assembler.transform(state_encoded)
    
    prediction = model.predict(assembled_df)   
    
    output = prediction

    return render_template('index.html',prediction_text= f"The value for Churn Prediction is {output}")

if __name__ == "__main__":
    app.run()