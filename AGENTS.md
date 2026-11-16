# AGENTS.md — ColabTensor

Mandatory rules for AI agents and developers. They are verified automatically by the tooling below.

## Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Language |
| RUFF | Linter + formatter (auto-verifies the rules in this file) |
| PyreFly | Static type checker |
| Pytest | Test runner |

## Naming Conventions

| Element | Style | Example |
|---|---|---|
| Folders | PascalCase | `DataModels/`, `ApplicationServices/` |
| Files | PascalCase | `UserService.py`, `OrderFactory.py` |
| Classes | PascalCase | `class OrderFactory:` |
| Exceptions | PascalCase + `Error` suffix | `class ValidationError(Exception):` |
| Variables | camelCase | `userName`, `totalCount` |
| Functions / methods | camelCase | `getTotalPrice()` |
| Parameters / arguments | camelCase | `def create(orderId):` |
| Attributes | camelCase | `self.orderCount` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRY = 3` |
| Private members | `_` prefix + camelCase | `_internalCache` |
| Tests | PascalCase | `Tests/TestUserService.py` |

## Architecture Rules

- Build with **classes and design patterns** (Factory, Strategy, Repository, Service, Builder, Adapter, Observer). No script-style modules with loose logic.
- **Restrict `if` usage**: no deep nesting, no long `if/elif` chains. Prefer polymorphism, dispatch tables, `match`/`case`, ternary expressions, and guard clauses with early returns.
- Write **structured code**: small functions, single responsibility, one abstraction level per function.
- Business logic never lives in `if __name__ == "__main__"` blocks or in tests.

## RUFF Auto-Verification

Config lives in `pyproject.toml` (`[tool.ruff]`). RUFF verifies these rules automatically:

| Requirement | RUFF rule |
|---|---|
| 80-char line length | `E501` + `line-length = 80` |
| PascalCase classes | `N801` |
| `Error` suffix on exceptions | `N818` |
| camelCase allowed for vars/functions | `N802`, `N803`, `N806`, `N815`, `N816` ignored |
| Restrict `if` / complexity | `PLR0911`, `PLR0912`, `PLR0913`, `PLR0914`, `PLR0915`, `C901` |
| Simplify conditionals | `SIM102`, `SIM103`, `SIM108` |
| No superfluous `else` | `RET505` |

Note: RUFF has no native camelCase checker. The PEP 8 snake_case rules (`N802`, `N803`, `N806`, `N815`, `N816`) are disabled so camelCase is accepted, while PascalCase classes stay enforced by `N801`. Agents must follow the camelCase convention above.

Commands:

```bash
ruff check .          # verify all rules
ruff check --fix .    # auto-fix
ruff format .         # format at 80 columns
```

## PyreFly

```bash
pyrefly init      # one-time setup (creates pyrefly.toml or updates pyproject.toml)
pyrefly check     # type-check the project
```

All public functions, methods, and attributes must be type-annotated.

## Pytest

Tests live in `Tests/` (PascalCase). Files: `Test*.py`, classes: `Test*`, functions: `test*` + camelCase.

```bash
pytest
```

## Definition of Done

A change is done only when all of these pass:

```bash
ruff check . && ruff format --check . && pyrefly check && pytest
```
