Demo of machine learning using Linear Regression

The app is currently deployed on Google Cloud: https://nyctaxi-457867227844.europe-west1.run.app/

The application use a data set of taxi trips in New York city containing 1000 rows of data such distance, pickup time, paid fare etc.
The ML linear regression model has been then trained on the data set using Databricks and the resulting model is deployed on Databricks with a REST api.

The second part of the demo is a user interface in the form of a web page that predicts the fare for a trip using the machine learning model through the REST api.


<img width="933" height="434" alt="Screenshot 2025-10-14 at 11 50 22" src="https://github.com/user-attachments/assets/2634753e-1298-455b-996a-3d5b0a5562f3" />


Example Databricks notebook run:

<img width="710" height="400" alt="Screenshot 2025-10-14 at 21 45 17" src="https://github.com/user-attachments/assets/6dae965c-f76b-4455-a319-64c94a134646" />

<img width="710" height="356" alt="Screenshot 2025-10-14 at 21 46 25" src="https://github.com/user-attachments/assets/836cab56-7308-4f4d-ae90-6cc193156fcc" />



Instructions
------------------------

- notebook.py file is the ML notebook uploaded to Databricks. It requires the test data file 'gkne-dk5s.csv' which should also be uploaded to Databricks.

- app directory contains the Pyton code for the front end as well as a Docker container for easy deployment to a cloud server like AWS or Google Cloud.

- set Databricks api key as environment variable 'DBRICKS_API' when running Docker container
