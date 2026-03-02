# Docstring standards

AceBet uses **Google-style docstrings** for modules, classes, and functions.

## What to include

- A concise first sentence describing behavior.
- `Args:` for important inputs.
- `Returns:` when returning values.
- `Raises:` for expected failure modes.

## Examples

### Function

```python
def query_data(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Filter rows for a specific date.

    Args:
        df: Match-level dataset.
        date: Date in ``YYYY-MM-DD`` format.

    Returns:
        pd.DataFrame: Rows matching the requested date.

    Raises:
        ValueError: If ``date`` cannot be parsed.
    """
```

### Class

```python
class Token(BaseModel):
    """Response model for bearer token issuance.

    Attributes:
        access_token: JWT token string.
        token_type: OAuth token type.
    """
```
