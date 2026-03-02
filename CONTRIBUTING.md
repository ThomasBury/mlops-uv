# Contributing

## Code quality expectations

- Keep functions, classes, and modules documented with **Google-style docstrings**.
- Prefer short behavioral summaries followed by parameter/return/raises contracts when applicable.
- Keep language technical and production-focused; avoid conversational or playful phrasing.

## Docstring convention (Google)

AceBet uses **Google-style docstrings** project-wide.

### Required shape

```python
def example(value: int) -> int:
    """Double an integer.

    Args:
        value: Input integer.

    Returns:
        int: Doubled integer value.

    Raises:
        ValueError: If ``value`` is negative.
    """
```

### Additional guidance

- Document `Args` for non-trivial parameters.
- Document `Returns` for non-`None` outputs.
- Include `Raises` when exceptions are part of normal contracts.
- Use one-sentence summaries for simple value objects where sections add no clarity.
