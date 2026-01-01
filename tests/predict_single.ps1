$body = @{
  pickup_lat = 40.758
  pickup_lon = -73.9855
  dropoff_lat = 40.7128
  dropoff_lon = -74.0060
  pickup_datetime = "2016-01-15T18:30:00"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body