# Contributing to EIE

## Adding a new scheduling strategy

1. Create a class implementing `PolicyStrategy` in `contrib/strategies/`
2. Implement all virtual methods
3. Register it in the factory or load via plugin

## Adding a GPU backend

1. Implement `ComputeBackend` in `contrib/backends/`
2. Add detection logic in `detectBackend()`

## Code style

- C++17, no exceptions in hot paths
- Use `std::cout` for info, `std::cerr` for errors
- All public headers in snake_case

## License

All contributions must be Apache 2.0 compatible.
