# Test Commands

## Run All Event-Related Tests

```bash
python3 -m unittest tests/test_event_round3.py tests/test_event_store.py -v
```

## Run Round-3 Event Logic Tests Only

```bash
python3 -m unittest tests/test_event_round3.py -v
```

## Run EventStore Tests Only

```bash
python3 -m unittest tests/test_event_store.py -v
```

## Expected Result

- Total tests: `13`
- Output ends with:

```text
Ran 13 tests in ...
OK
```
