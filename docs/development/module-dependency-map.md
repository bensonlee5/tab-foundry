# Module Dependency Map

This is the intended dependency direction for `tab-foundry`.

It is a planning document for now, not an enforced architecture check.

## Intended Direction

```text
cli/commands -> cli -> bench
cli/commands -> training/data/model/export
bench -> training/data/model/export
training -> model/data
export -> model
model/architectures -> model/components
data/sources -> data
```

## Notes

- `bench/` may depend on core library packages, but core library packages should not depend on `bench/`.
- `cli/commands/` may orchestrate both `bench/` and core packages.
- `model/components/` should remain reusable and family-agnostic.
- `model/architectures/` should assemble components into named families.
- current export compatibility constraints around `tabiclv2` are tolerated at the boundary, but should not re-anchor internal architecture structure.

## Future Follow-Up

Once the structural refactor lands, this map should become precise enough to support automated dependency checks similar in spirit to the structural enforcement work in `dagzoo`.
