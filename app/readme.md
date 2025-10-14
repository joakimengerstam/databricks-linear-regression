
# Build for local testing on osx
docker build -t joaeng/nyctaxi .

# Build for deployment on Google Cloud amd64 architecture
docker build --platform linux/amd64 -t joaeng/nyctaxi:amd64 . 

# Push to Docker hub
docker push joaeng/nyctaxi:amd64

Docker hub url:
docker.io/joaeng/nyctaxi:amd64


# Test if databricks endpoint is reachable from inside container
docker exec -it suspicious_johnson /bin/bash

curl -X POST https://dbc-a4864acb-d90e.cloud.databricks.com/serving-endpoints/nyc-taxi-fare/invocations \
  -H "Authorization: Bearer dapi252f160d2af154aa8f13a166fbbcc55d" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split":{"columns":["trip_distance","is_rush_hour"],"data":[[2.5,1]]}}'



