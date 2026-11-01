# FastAPI

## Dependencies

FastAPI dependencies are callables declared with `Depends`. FastAPI resolves them
for each request and injects their return values into endpoint parameters.

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def get_current_user():
    return {"name": "Ada"}

@app.get("/users/me")
def read_current_user(user: dict = Depends(get_current_user)):
    return user
```

Dependencies can themselves declare dependencies. This is useful for sharing
database sessions, authentication checks, and common request logic.

## Request validation

FastAPI uses Pydantic models to parse and validate request bodies. Invalid input
is rejected with an HTTP 422 response by default.

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
```

## Error responses

Raise `HTTPException` to return an HTTP error response from an endpoint. Its
`status_code` and `detail` are included in the response.

```python
from fastapi import HTTPException

raise HTTPException(status_code=404, detail="Item not found")
```
