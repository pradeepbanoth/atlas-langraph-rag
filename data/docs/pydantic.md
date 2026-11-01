# Pydantic

## Models and field validation

A Pydantic `BaseModel` validates data when it is created. Type annotations
describe the expected values, while `Field` adds constraints such as minimum and
maximum lengths.

```python
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    age: int = Field(ge=0)
```

## Custom validators

Use `field_validator` when validation needs logic beyond built-in constraints.
An after-style validator can normalize a value and raise `ValueError` to reject
invalid data.

```python
from pydantic import BaseModel, field_validator

class SearchRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value
```

## Settings from environment

`pydantic-settings` provides `BaseSettings`, which reads configuration from
environment variables. A settings model can also load a local `.env` file.
