# Complete naming conventions and repository best practices cheat sheet

This guide consolidates authoritative conventions for JAX, Python, configuration formats, and repository structure into actionable do/don't patterns. **Consistency within your project matters more than any specific convention**—choose patterns, document them, and enforce them.

---

## JAX-specific naming conventions

JAX's functional paradigm requires distinct patterns from standard Python, particularly around immutability, explicit randomness, and transform composition.

### PRNG key naming

The **key/subkey pattern** is JAX's canonical approach to explicit random state management. Keys that will be split further are named `key`; keys consumed immediately are `subkey`.

```python
# ✅ DO: Use key/subkey pattern
key = jax.random.key(42)  # New typed key (preferred since JAX 0.4.16)
key, subkey = jax.random.split(key)
samples = jax.random.normal(subkey, shape=(100,))

# ✅ DO: Purpose-specific key names for clarity
init_key, dropout_key, sampling_key = jax.random.split(key, 3)

# ❌ DON'T: Reuse keys (produces identical values)
key = jax.random.key(42)
a = jax.random.normal(key)
b = jax.random.normal(key)  # Identical to 'a'!

# ❌ DON'T: Use legacy PRNGKey
key = jax.random.PRNGKey(42)  # Deprecated; use jax.random.key()
```

**Rationale:** JAX's splittable PRNG requires explicit state threading. The key/subkey convention makes consumption tracking obvious—subkeys are "spent" immediately while the parent key remains for future splits.

### Import conventions and array operations

```python
# ✅ DO: Standard import aliases
import jax
import jax.numpy as jnp
import numpy as np  # or 'onp' for "original numpy" clarity

# ✅ DO: Functional array updates
arr = jnp.zeros(10)
arr = arr.at[0].set(5)
arr = arr.at[1:3].add(10)

# ❌ DON'T: In-place mutation (will fail or produce incorrect gradients)
arr[0] = 5
arr += 1
```

### Transform naming patterns

When composing transforms like `jit`, `vmap`, and `grad`, use descriptive names that convey the transformation applied.

```python
# ✅ DO: Compose transforms with clear naming
@jax.jit
def train_step(params, batch):
    def loss_fn(p):  # Inner functions often use _fn suffix
        return compute_loss(p, batch)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

# ✅ DO: Descriptive names for transformed functions
batched_predict = jax.vmap(predict, in_axes=(None, 0))  # Batch over second arg
loss_and_grad = jax.value_and_grad(loss_fn)
parallel_train = jax.pmap(train_step, axis_name='devices')

# ❌ DON'T: Generic names that obscure transform intent
f2 = jax.jit(jax.vmap(f))  # What does f2 do?
```

### Parameter tree structure

JAX ecosystem libraries use nested dictionaries for parameters. Standard leaf names are `kernel` (Flax), `w` (Haiku), or `weight` (Equinox) for weight matrices.

```python
# ✅ DO: Standard Flax parameter structure
params = {
    'params': {
        'dense_0': {'kernel': array(...), 'bias': array(...)},
        'dense_1': {'kernel': array(...), 'bias': array(...)},
    }
}

# ✅ DO: Use collection names for different parameter types
variables = {
    'params': {...},           # Trainable parameters
    'batch_stats': {...},      # BatchNorm statistics
    'cache': {...},            # Attention KV cache
}

# Flax module naming (auto-generates as ClassName_index)
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)          # Auto-named 'Dense_0'
        x = nn.Dense(64, name='out')(x)  # Explicit name 'out'
```

### Shape annotations with jaxtyping

The `jaxtyping` library provides semantic dimension names for array shapes, making code self-documenting.

```python
from jaxtyping import Array, Float, Int, PRNGKeyArray

# ✅ DO: Semantic dimension names
def attention(
    query: Float[Array, "batch heads seq_q dim"],
    key: Float[Array, "batch heads seq_k dim"],
    value: Float[Array, "batch heads seq_k dim_v"],
) -> Float[Array, "batch heads seq_q dim_v"]:
    ...

# ✅ DO: Use modifiers for flexibility
def process(x: Float[Array, "*batch channels"]) -> Float[Array, "*batch"]:
    ...  # *batch = zero or more batch dimensions

# Common dimension names: batch, seq, time, channels, height, width, features, heads
```

---

## Python naming conventions

### Core PEP 8 patterns

| Entity | Convention | Example |
|--------|------------|---------|
| Variables, functions | `snake_case` | `user_count`, `get_user()` |
| Classes | `PascalCase` | `UserAccount`, `HTTPServer` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_CONNECTIONS`, `PI` |
| Modules | `lowercase` | `my_module.py` |
| Packages | `lowercase` (no underscores) | `mypackage` |

```python
# ✅ DO
user_count = 42
def calculate_average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

class HTTPServerError(Exception):  # Acronyms stay uppercase
    pass

MAX_RETRY_ATTEMPTS = 3

# ❌ DON'T
UserCount = 42  # PascalCase for variable
def CalculateAverage(Numbers): ...  # PascalCase for function
class Http_Server_Error: ...  # Mixed conventions
```

### Private members and name mangling

Python uses underscore prefixes as visibility conventions, not enforcement mechanisms.

```python
# ✅ DO: Single underscore for internal use
class APIClient:
    def __init__(self):
        self._session = None      # Internal, but accessible
        self._cache = {}
    
    def _refresh_token(self):     # Internal method
        ...

# ✅ DO: Single trailing underscore to avoid keyword conflicts
def create_element(class_='div', type_='text'):
    ...

# ✅ DO: Double underscore ONLY for subclass conflict prevention
class Parent:
    def __init__(self):
        self.__id = uuid4()  # Mangled to _Parent__id

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.__id = uuid4()  # Mangled to _Child__id (no collision)

# ❌ DON'T: Double underscore just to "hide" attributes
class MyClass:
    def __init__(self):
        self.__data = []  # Overkill—use self._data instead
```

**Rationale:** Double underscore triggers name mangling (`__attr` becomes `_ClassName__attr`), which is designed specifically for inheritance hierarchies where attribute name collision is a real concern. For simple "internal use" signaling, single underscore is clearer and more Pythonic.

### Type hints and generics

Modern Python (3.10+) uses built-in generics and union syntax. TypeVars use short `CapWords` names.

```python
from typing import TypeVar, Protocol, TypeAlias

# ✅ DO: Short names for unconstrained TypeVars
T = TypeVar('T')
K = TypeVar('K')  # Key
V = TypeVar('V')  # Value

# ✅ DO: Descriptive names for constrained TypeVars
Numeric = TypeVar('Numeric', int, float, complex)
T_co = TypeVar('T_co', covariant=True)  # _co/_contra suffixes for variance

# ✅ DO: Modern syntax (Python 3.10+)
def process(items: list[str | int]) -> dict[str, Any]:  # Built-in generics
    ...

Vector: TypeAlias = list[float]  # Type alias with annotation

# ✅ DO: Protocol naming with -able or -Protocol suffix
class Drawable(Protocol):
    def draw(self) -> None: ...

class SupportsRead(Protocol):
    def read(self, size: int = -1) -> bytes: ...

# ❌ DON'T: Legacy typing imports (Python 3.9+)
from typing import List, Dict, Optional  # Use list, dict, X | None instead
```

### Exception and decorator naming

```python
# ✅ DO: Exception classes end with 'Error' (for actual errors)
class ValidationError(Exception):
    """Raised when input validation fails."""

class DatabaseConnectionError(Exception):
    """Raised when database connection cannot be established."""

# ✅ DO: Non-error exceptions use descriptive names without 'Error'
class StopProcessing(Exception):  # Flow control signal
    pass

# ✅ DO: Decorators use snake_case, verb-like names
def retry(max_attempts: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ...
        return wrapper
    return decorator

@retry(max_attempts=5)
@validate_input
def fetch_data(url: str) -> dict:
    ...
```

### Async function naming

Two conventions exist—choose based on codebase context:

```python
# ✅ DO: No prefix in fully async codebases
async def fetch_user(user_id: int) -> User:
    async with session.get(f"/users/{user_id}") as resp:
        return await resp.json()

# ✅ DO: 'a' prefix for dual sync/async APIs (Django pattern)
class QuerySet:
    def get(self, **kwargs): ...        # Sync
    async def aget(self, **kwargs): ... # Async variant
    
    def count(self): ...
    async def acount(self): ...

# ✅ DO: '_async' suffix as alternative
def connect(host: str) -> Connection: ...
async def connect_async(host: str) -> Connection: ...
```

---

## Configuration and infrastructure naming

### Environment variables

The **SCREAMING_SNAKE_CASE** convention is universal for environment variables, with application prefixes preventing collisions.

```bash
# ✅ DO: Uppercase with underscores, application prefix
MYAPP_DATABASE_HOST=localhost
MYAPP_DATABASE_PORT=5432
MYAPP_REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY_ID=AKIA...
LOG_LEVEL=INFO

# ✅ DO: Hierarchical naming with double underscore (some frameworks)
MYAPP__DATABASE__HOST=localhost
MYAPP__DATABASE__PORT=5432

# ❌ DON'T
database-host=localhost     # Hyphens not portable
myapp.db.host=localhost     # Dots problematic in many shells
DatabaseHost=localhost      # Case inconsistency
```

### YAML and TOML keys

Convention depends on ecosystem—Python tools typically use `snake_case`, Kubernetes uses `camelCase`.

```yaml
# ✅ DO: Python ecosystem (Ansible, Docker Compose) - snake_case
database:
  connection_pool_size: 10
  max_idle_time_seconds: 300

# ✅ DO: Kubernetes ecosystem - camelCase
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: app
      containerPort: 8080
```

```toml
# ✅ DO: pyproject.toml follows PEP 621 - kebab-case for keys
[project]
name = "my-project"
requires-python = ">=3.10"
dependencies = ["requests>=2.0"]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[tool.ruff]
line-length = 100
```

### CLI arguments

GNU/POSIX conventions: single hyphen for single-character flags, double hyphen for long flags with **kebab-case**.

```bash
# ✅ DO: Standard flag patterns
myapp -v                    # Short flag
myapp --verbose             # Long flag equivalent
myapp -o output.txt         # Short with value
myapp --output=output.txt   # Long with value
myapp --dry-run             # kebab-case for multi-word
myapp --no-cache            # Boolean negation with no- prefix

# ✅ DO: Common conventions
-h, --help          # Always provide
-v, --verbose       # Increase verbosity
-q, --quiet         # Decrease verbosity  
-V, --version       # Show version
-n, --dry-run       # Preview without executing

# ❌ DON'T
--dryRun            # camelCase
--dry_run           # snake_case
--DRY-RUN           # UPPERCASE
```

### Database naming

PostgreSQL and most databases prefer **snake_case** due to case-folding behavior. Use plural table names to represent collections.

```sql
-- ✅ DO: snake_case, plural tables, descriptive columns
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email_address VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),  -- FK: {table}_id
    quantity INTEGER NOT NULL
);

-- ✅ DO: Consistent constraint naming
CREATE INDEX idx_users_email ON users(email_address);
ALTER TABLE orders ADD CONSTRAINT orders_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(user_id);

-- ❌ DON'T: Mixed case requires quoting forever
CREATE TABLE "Users" ("UserId" SERIAL);  -- Must quote in all queries
```

**Migration file naming:** Use timestamp prefix with snake_case description.
```
20250102143022_create_users_table.sql
20250103091500_add_email_index_to_users.sql
```

### REST API endpoints

Use **plural nouns** for collections, **kebab-case** for multi-word paths, and let HTTP methods convey actions.

```
# ✅ DO: RESTful resource naming
GET    /users                  # List collection
POST   /users                  # Create resource
GET    /users/{id}             # Get specific resource
PUT    /users/{id}             # Replace resource
PATCH  /users/{id}             # Partial update
DELETE /users/{id}             # Delete resource

GET    /users/{id}/orders      # Nested collection
GET    /user-profiles          # kebab-case for multi-word

# ✅ DO: Versioning in path
GET    /v1/users
GET    /v2/users

# ❌ DON'T: Verbs in URLs (HTTP method conveys action)
GET    /getUsers
POST   /createUser
PUT    /updateUser/{id}
DELETE /deleteUser/{id}

# ❌ DON'T: Inconsistent casing
GET    /Users                  # Should be lowercase
GET    /user_profiles          # Should be kebab-case
```

**JSON response fields:** Use `camelCase` for JavaScript-facing APIs or `snake_case` for Python-native backends—be consistent.

```json
{
  "userId": 123,
  "firstName": "Jane",
  "createdAt": "2025-01-02T10:00:00Z",
  "orderItems": []
}
```

---

## Repository structure and quality practices

### Project layout

The **src layout** is recommended by PyPA for distributable packages—it prevents accidental imports of development code over installed packages.

```
my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   └── test_core.py
│   └── integration/
│       └── test_api.py
├── docs/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── LICENSE
```

**Standard directories:**
- `src/` — Importable package code only
- `tests/` — Test files (outside package, avoids shipping tests)
- `docs/` — Documentation source files
- `scripts/` — Utility and maintenance scripts
- `.github/` — GitHub Actions, issue templates, PR templates

### Git commit messages

**Conventional Commits** format enables automated changelog generation and semantic versioning.

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

```bash
# ✅ DO: Conventional commit format
feat(auth): add OAuth2 login support
fix(api): prevent race condition in request handling
docs(readme): add installation instructions
refactor(core): simplify state management
test(auth): add integration tests for login flow

# ✅ DO: Breaking changes with ! or footer
feat!: change API response format

BREAKING CHANGE: Response now wraps data in 'result' key

# ❌ DON'T
Fixed stuff                           # Too vague
feat: Add feature                     # Capitalize after colon
updated code, fixed bugs, added tests # Multiple concerns
```

| Type | Purpose | SemVer Impact |
|------|---------|---------------|
| `feat` | New feature | MINOR |
| `fix` | Bug fix | PATCH |
| `docs` | Documentation | None |
| `refactor` | Code restructure | None |
| `test` | Test changes | None |
| `chore` | Maintenance | None |
| `perf` | Performance | PATCH |
| `ci` | CI/CD changes | None |

**Subject line rules:** Imperative mood ("add" not "added"), no period, max **50 characters** (72 hard limit).

### Branch naming

Use lowercase **kebab-case** with type prefixes. Include ticket numbers when applicable.

```bash
# ✅ DO
feature/add-user-authentication
feature/PROJ-123-oauth-integration
bugfix/fix-login-timeout
hotfix/critical-security-patch
release/v1.2.0
docs/update-api-reference

# ❌ DON'T
johns-stuff                # Not descriptive
Feature/AddLogin           # Wrong case
feature_add_login          # Underscores
```

### Testing conventions

pytest discovers tests by naming convention—files must match `test_*.py` or `*_test.py`, functions must start with `test_`.

```python
# File: tests/unit/test_authentication.py

import pytest

# ✅ DO: Fixtures use descriptive names (no prefix required)
@pytest.fixture
def authenticated_user():
    return User(id=1, email="test@example.com")

@pytest.fixture
def mock_database(mocker):
    return mocker.patch("app.db.connection")

# ✅ DO: Test functions describe behavior
def test_login_returns_token_when_credentials_valid(authenticated_user):
    result = login(authenticated_user.email, "password")
    assert result.token is not None

def test_login_raises_error_when_password_invalid():
    with pytest.raises(AuthenticationError):
        login("user@example.com", "wrong")

# ✅ DO: Test classes group related tests (no __init__)
class TestUserAuthentication:
    def test_successful_login(self): ...
    def test_failed_login_wrong_password(self): ...
    def test_account_locked_after_attempts(self): ...

# ❌ DON'T: These won't be discovered
def check_login(): ...           # Missing test_ prefix
def testLogin(): ...             # Needs underscore: test_login
class UserTests: ...             # Needs Test prefix: TestUser
```

### Versioning and changelog

**Semantic Versioning (SemVer):** `MAJOR.MINOR.PATCH` where MAJOR = breaking changes, MINOR = new features, PATCH = bug fixes.

```markdown
# CHANGELOG.md (Keep a Changelog format)

## [Unreleased]

### Added
- OAuth2 authentication support

## [1.2.0] - 2025-01-02

### Added
- User profile endpoints (#123)
- Rate limiting middleware

### Fixed
- Connection timeout on slow networks (#456)

### Deprecated
- Legacy /v1/auth endpoints (use /v2/auth)

## [1.1.0] - 2024-12-15
...
```

**Change categories:** Added, Changed, Deprecated, Removed, Fixed, Security

### Pre-commit configuration

Essential hooks for Python projects using modern tooling:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff          # Linting (replaces Flake8)
        args: [--fix]
      - id: ruff-format   # Formatting (replaces Black)

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
```

Run `pre-commit install` after cloning and `pre-commit autoupdate` periodically to keep hooks current.

---

## Quick reference table

| Context | Convention | Example |
|---------|------------|---------|
| Python variables/functions | `snake_case` | `user_count`, `get_user()` |
| Python classes | `PascalCase` | `UserAccount` |
| Python constants | `SCREAMING_SNAKE_CASE` | `MAX_RETRIES` |
| Python private | `_single_underscore` | `_internal_cache` |
| TypeVars | Short `CapWords` | `T`, `K`, `V` |
| JAX imports | `jnp`, `np` | `import jax.numpy as jnp` |
| JAX PRNG | `key`/`subkey` | `key, subkey = jax.random.split(key)` |
| Environment vars | `SCREAMING_SNAKE_CASE` | `DATABASE_URL` |
| CLI flags | `-x`, `--kebab-case` | `-v`, `--dry-run` |
| Database tables | `snake_case`, plural | `users`, `order_items` |
| Database columns | `snake_case`, singular | `user_id`, `created_at` |
| REST endpoints | lowercase, kebab-case | `/v1/user-profiles` |
| JSON keys | `camelCase` or `snake_case` | `userId` or `user_id` |
| Git branches | `type/kebab-case` | `feature/add-auth` |
| Git commits | Conventional Commits | `feat(auth): add OAuth` |
| Test files | `test_*.py` | `test_authentication.py` |
| Test functions | `test_*` | `test_login_succeeds()` |

---

## Conclusion

Three principles govern effective naming conventions: **consistency** (one pattern per project), **clarity** (names reveal intent), and **convention alignment** (follow ecosystem norms). For JAX code, embrace functional patterns with explicit key management and transform composition. For Python, PEP 8 remains authoritative but modern type hints and union syntax (`X | None`) reflect current best practice. Repository structure benefits from the src layout, Conventional Commits, and automated tooling via pre-commit hooks. When conventions conflict across ecosystems—such as `camelCase` JSON for JavaScript consumers versus `snake_case` for Python backends—document the choice and enforce it consistently.