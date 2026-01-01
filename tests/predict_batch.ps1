$body = @{
  trips = @(
    @{
      pickup_lat = 40.758
      pickup_lon = -73.9855
      dropoff_lat = 40.7128
      dropoff_lon = -74.0060
      pickup_datetime = "2016-01-15T18:30:00"
    },
    @{
      pickup_lat = 40.73061
      pickup_lon = -73.935242
      dropoff_lat = 40.650002
      dropoff_lon = -73.949997
      pickup_datetime = "2016-01-15T08:15:00"
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict_batch" -Method Post -ContentType "application/json" -Body $body