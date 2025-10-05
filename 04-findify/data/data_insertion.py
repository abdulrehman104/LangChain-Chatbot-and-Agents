# import library
import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_PASSWORD = os.getenv('DB_PASSOWARD')
print(f"DB_PASSWORD: {DB_PASSWORD}")

# read csv file:
csv_file='shop-product-catalog.csv'
data=pd.read_csv(csv_file)

# connect to MySQL
db_connection=mysql.connector.connect(
    host='localhost', 
    user='root', 
    password=DB_PASSWORD, 
    database='findify'
    )

cursor=db_connection.cursor()

# upload data to MySQL
for index,row in data.iterrows():
    sql="""
    INSERT INTO products (ProductID,ProductName,ProductBrand,Gender,Price,Description,PrimaryColor)
    VALUES (%s, %s,%s,%s,%s,%s,%s)
    """
    cursor.execute(sql,tuple(row))

db_connection.commit()


cursor.close()
db_connection.close()