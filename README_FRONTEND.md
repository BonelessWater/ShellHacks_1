Frontend integration quick-start

Fetch invoices from the backend

Example GET to fetch invoices (with pagination and simple filters):

```
GET http://localhost:8000/api/invoices?limit=50&offset=0&vendor=Acme
```

Example response (frontend-friendly shape):

{
  "invoices": [
    {
      "id": "INV-2024-001",
      "vendor": {"name":"ABC Office Supplies","address":"","phone":"","email":"","tax_id":""},
      "amount": 1287.00,
      "status": "approved",
      "confidence": 0.95,
      "issues": 0,
      "date": "2024-01-15",
      "description": "Monthly office supplies",
      "line_items": [
        {"description":"Paper","quantity":10,"unit_price":12.87,"total":128.7,"sku":""}
      ],
      "_raw": { /* original record from BigQuery or backend */ }
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0,
  "has_more": false
}

Notes:
- The `vendor` field is normalized into an object with common fields.
- `line_items` is always an array of objects (may be empty).
- `_raw` contains the original backend/bigquery record for advanced UIs.
