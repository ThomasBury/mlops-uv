# Docstring standards

AceBet uses **NumPy/SciPy-style docstrings** for modules, classes, and
functions.

This matches the current codebase. When editing existing docstrings, keep that
style instead of mixing Google-style sections into the same repository.

## What to include

- A concise first sentence describing behavior.
- `Parameters` for important inputs.
- `Returns` when returning values.
- `Raises` for expected failure modes.
- `Attributes` for data containers when the extra detail is useful.

## Examples

### Function

```python
def query_data(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Filter rows for a specific date.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level dataset.
    date : str
        Date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Rows matching the requested date.

    Raises
    ------
    ValueError
        If ``date`` cannot be parsed.
    """
```

### Class

```python
class Token(BaseModel):
    """Response model for bearer token issuance.

    Attributes
    ----------
    access_token : str
        JWT token string.
    token_type : str
        OAuth token type.
    """
```
