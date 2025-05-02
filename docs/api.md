# API Documentation

## Endpoints

### Health Check
```http
GET /health
```
Returns the health status of the API.

### Analyze Image
```http
POST /api/v1/analyze
```
Analyzes an image for sutures.

#### Request
- Content-Type: `multipart/form-data`
- Body: 
  - `file`: Image file (PNG, JPG, JPEG)

#### Response
```json
{
  "results": [
    {
      "box": [x1, y1, x2, y2],
      "score": 0.95,
      "class_id": 0
    }
  ]
}
```

## Error Handling

All endpoints return standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 500: Internal Server Error

Error responses include a message:
```json
{
  "error": "Error description"
}
```