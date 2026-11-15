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

```python title="Function docstring example"
def load_player_stats(player_stats_file: str | Path) -> pd.DataFrame:
    """Load the online player-state table.

    Parameters
    ----------
    player_stats_file : str | Path
        Feather file containing one current snapshot per player.

    Returns
    -------
    pd.DataFrame
        Player-state rows indexed by player.

    Raises
    ------
    ValueError
        If the table cannot be read or has invalid state columns.
    """
```

### Class

```python title="Class docstring example"
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
