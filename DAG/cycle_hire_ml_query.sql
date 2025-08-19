-- SQL query to extract features for predicting bike trip duration
-- Target: trip duration in minutes
-- Features: day of week, hour of day, start station, end station, bike type

CREATE OR REPLACE TABLE `YOUR_PROJECT_ID.ml_dataset.cycle_hire_features` AS
SELECT 
  trip_id,
  EXTRACT(DAYOFWEEK FROM start_date) as day_of_week,
  EXTRACT(HOUR FROM start_date) as hour_of_day,
  start_station_id,
  end_station_id,
  bike_id,
  duration / 60 as trip_duration_minutes,  -- Target variable
  ST_DISTANCE(
    ST_GEOGPOINT(start_station_longitude, start_station_latitude),
    ST_GEOGPOINT(end_station_longitude, end_station_latitude)
  ) as distance_meters
FROM `bigquery-public-data.london_bicycles.cycle_hire`
WHERE 
  duration IS NOT NULL 
  AND duration > 0 
  AND duration < 86400  -- Filter out trips longer than 24 hours
  AND start_station_id IS NOT NULL 
  AND end_station_id IS NOT NULL
  AND start_date >= '2015-01-01'
  AND start_date < '2016-01-01'  -- Use one year of data
LIMIT 10000  -- Keep it simple for demo