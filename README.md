## Run

### python
```bash
python main.py
```

### spark-submit
```bash
$ ./bin/spark-submit ~/main.py
```


```python
import os
import re
os.environ["JAVA_HOME"] = "/home/zulujava8"
os.environ["SPARK_HOME"] = "/home/spark-3.3.1-bin-hadoop3"

from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, udf
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pyspark



class prediction:
    def __init__(self, modelpath: str):
        self.spark = SparkSession.builder\
            .master("local[1]")\
            .appName("nlp_inference")\
            .config("spark.ui.port", "4050")\
            .config("spark.driver.memory", "1g")\
            .getOrCreate()
        self.model = PipelineModel.load(modelpath)

    def loadCsv(self, filepath: str) -> pyspark.sql.dataframe.DataFrame:
        df = (
            self.spark.read.format("csv")
            .option("header", "true")
            .option("sep", ",")
            .load(filepath)
        )

        def removeSpecialChar(v: str) -> str:
            x = re.sub("[^A-Za-z0-9 ]+", " ", v)
            return x.lower()

        def removeDoubleSpace(v: str) -> str:
            x = re.sub("[ ]{2,}", " ", v)
            return x

        udfReSpeChar = udf(removeSpecialChar, StringType())
        udfReSpace = udf(removeDoubleSpace, StringType())

        df = df.withColumn("clean_text", udfReSpeChar("text"))
        df = df.withColumn("clean_text", udfReSpace("clean_text"))
        return df

    def doPrediction(self, df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
        def labelling(val: float) -> str:
            if val == 0.0:
                return "positive"
            elif val == 1.0:
                return "negative"
            elif val == 2.0:
                return "neutral"
        
        result = self.model.transform(df)
        udfLabelling = udf(labelling, StringType())

        result = result.withColumn(
            "predictionLabel", udfLabelling("prediction"))
        print(result.show(3))
        return result

    def exportResult(self, df: pyspark.sql.dataframe.DataFrame):
        df.write.mode("overwrite").parquet("./result_parquet")
        
        df.toPandas()\
            .to_json(orient="records", path_or_buf="./result_full_json.json")
        
        df.select(["text","clean_text","probability","prediction","predictionLabel"]).toPandas()\
            .to_json(orient="records", path_or_buf="./result_easy_json.json")

    def stopSpark(self):
        self.spark.stop()


if __name__ == "__main__":
    p = prediction("./nlp_simple")
    df = p.loadCsv("./text.csv")
    df = p.doPrediction(df)
    p.exportResult(df)
    p.stopSpark()
```

### Output
```bash
+--------------------+--------------------+--------------------+----------+---------------+  
|                text|          clean_text|         probability|prediction|predictionLabel|  
+--------------------+--------------------+--------------------+----------+---------------+  
|kemarin gue datan...|kemarin gue datan...|[0.43664724613529...|       2.0|        neutral|  
|kayak nya sih gue...|kayak nya sih gue...|[0.01089547459217...|       2.0|        neutral|  
|kalau dipikir-pik...|kalau dipikir pik...|[0.01126194570294...|       2.0|        neutral|  
+--------------------+--------------------+--------------------+----------+---------------+
only showing top 3 rows
```
