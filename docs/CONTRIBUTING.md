# Contributing to EIE

## Adding a new scheduling strategy

1. Create a class implementing `PolicyStrategy` in `contrib/strategies/`
2. Implement all virtual methods
3. Register it in the factory. Dynamic plugin loading is not implemented.

## Claims and evidence

Describe code presence, local execution, third-party replication and planned
behavior separately. A passing unit test is not a complete deployment test.
Performance comparisons need matched artifacts/settings and retained raw
measurements. Preserve failed cases and frozen artifacts; qualify historical
claims with a scope note rather than changing their original evidence.

See [CLAIMS_AUDIT.md](CLAIMS_AUDIT.md) and
[ROADMAP_TO_CLAIMS.md](ROADMAP_TO_CLAIMS.md).

## Adding a GPU backend

1. Implement `ComputeBackend` in `contrib/backends/`
2. Add detection logic in `detectBackend()`

## Code style

- C++17, no exceptions in hot paths
- Use `std::cout` for info, `std::cerr` for errors
- All public headers in snake_case

## License

All contributions must be Apache 2.0 compatible.
