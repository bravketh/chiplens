# Validation

Published results with sources live in `data/`. `results.md` is generated:

```bash
chiplens validate --write
```

A test fails if `results.md` is out of date. Current finding: chiplens does not
reproduce the SN40L Table IV numbers (errors +73%, -55%, -72%), and the report says why.
