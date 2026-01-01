$body = @{
  trips = @(
    @{
      pickup_lat = 400
      pickup_lon = -73.9
      dropoff_lat = 40.7
      dropoff_lon = -74.0
      pickup_datetime = "not-a-date"
    }
  )
} | ConvertTo-Json -Depth 5

try {
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict_batch" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
}
catch {
  Write-Host "Validation error caught as expected"
  Write-Host $_.ErrorDetails.Message
}