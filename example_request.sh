#!/bin/bash

# Base URL
BASE_URL="http://localhost:8000"

echo "=== Health Check ==="
curl -X GET "$BASE_URL/api/health"

echo -e "\n\n=== Product Query ==="
curl -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123", 
    "query": "Do you have Nike Air Max shoes in size 42?"
  }'

echo -e "\n\n=== General Greeting ==="
curl -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_456", 
    "query": "Hello! How are you?"
  }'

echo -e "\n\n=== Price Query ==="
curl -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_789", 
    "query": "What are the prices of Adidas running shoes?"
  }'