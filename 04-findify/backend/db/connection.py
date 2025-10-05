import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

db_passooward = os.getenv("DB_PASSOWARD")
print("Database password:", db_passooward)

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password=db_passooward,
        database='findify'
    )

# db = get_db_connection()
# print("Database connection established:", db.is_connected())