# Parser Test Document

This fixture checks that headings inside fenced code are ignored.

## Valid Section

Normal paragraph before the code sample.

```python
# This is a comment, not a heading
## neither is this
def parse():
    return {"title": "## fake"}
```

## Another Valid Section

Inline `## backticks` should not create a node either.

```markdown
# README inside code fence
## Installation
```

## Final Section

End of document.
