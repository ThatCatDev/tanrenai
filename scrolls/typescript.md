---
name: typescript
description: TypeScript coding standards using OOP with classes, SOLID principles, and design patterns for modular architecture
tags: [typescript, ts, react, nextjs, node, component, module, solid, design-pattern, frontend, backend, api, service, oop, class]
---

# TypeScript Coding Standards

## Core Philosophy: OOP with Classes

**Always use classes.** Structure all business logic, services, repositories, controllers, and domain entities as classes. Use interfaces to define contracts, classes to implement them. No loose functions for anything beyond simple utilities.

## Architecture: Modular & Component-Based

Every class and module must be **self-contained and self-sufficient**. A module should be usable without knowing anything about the rest of the system.

### Directory Structure

```
src/
├── components/        # UI components (if frontend)
│   └── Button/
│       ├── Button.tsx
│       ├── Button.test.tsx
│       ├── Button.styles.ts
│       └── index.ts
├── modules/           # Feature modules
│   └── auth/
│       ├── auth.controller.ts
│       ├── auth.service.ts
│       ├── auth.repository.ts
│       ├── auth.types.ts
│       ├── auth.errors.ts
│       ├── auth.test.ts
│       └── index.ts
├── shared/            # Cross-cutting concerns
│   ├── types/
│   ├── errors/
│   └── base/          # Abstract base classes
│       ├── base.service.ts
│       ├── base.repository.ts
│       └── base.controller.ts
└── index.ts           # Public API barrel
```

**Rules:**
- Each module folder has its own `index.ts` that exports the public API. Nothing outside the folder imports from internal files directly.
- Co-locate tests, types, and styles with the code they belong to.
- No circular dependencies between modules. Use dependency injection instead.

## SOLID Principles

### S — Single Responsibility
Each class does exactly one thing. One reason to change.

```typescript
// BAD: one class does everything
class UserManager {
  async getUser(id: string) { /* db query */ }
  toJSON(user: User) { /* serialization */ }
  sendWelcomeEmail(user: User) { /* email logic */ }
}

// GOOD: separate classes, separate concerns
class UserRepository {
  async findById(id: string): Promise<User> { ... }
}

class UserSerializer {
  toDto(user: User): UserDto { ... }
}

class WelcomeEmailService {
  async send(user: User): Promise<void> { ... }
}
```

### O — Open/Closed
Extend behavior through new classes, not by modifying existing ones.

```typescript
// Define the contract
interface NotificationChannel {
  send(to: string, message: string): Promise<void>;
}

// Extend by adding new classes
class EmailChannel implements NotificationChannel {
  constructor(private readonly mailer: Mailer) {}
  async send(to: string, message: string): Promise<void> { ... }
}

class SlackChannel implements NotificationChannel {
  constructor(private readonly webhook: string) {}
  async send(to: string, message: string): Promise<void> { ... }
}

// Consumer depends on the interface, not the implementations
class NotificationService {
  constructor(private readonly channels: NotificationChannel[]) {}

  async notify(to: string, message: string): Promise<void> {
    await Promise.all(this.channels.map(ch => ch.send(to, message)));
  }
}
```

### L — Liskov Substitution
Subtypes must be substitutable for their base types without breaking behavior.

```typescript
abstract class Cache<T> {
  abstract get(key: string): Promise<T | null>;
  abstract set(key: string, value: T, ttl?: number): Promise<void>;
  abstract delete(key: string): Promise<void>;
}

// Both honor the same contract — callers can swap freely
class RedisCache<T> extends Cache<T> { ... }
class InMemoryCache<T> extends Cache<T> { ... }
```

### I — Interface Segregation
Don't force classes to implement methods they don't need.

```typescript
// BAD: fat interface
interface Repository<T> {
  find(id: string): Promise<T>;
  findAll(): Promise<T[]>;
  create(data: T): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
  bulkInsert(data: T[]): Promise<T[]>;
  aggregate(pipeline: any): Promise<any>;
}

// GOOD: segregated interfaces
interface Readable<T> {
  find(id: string): Promise<T>;
  findAll(): Promise<T[]>;
}

interface Writable<T> {
  create(data: T): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

// Classes compose only the interfaces they need
class UserRepository implements Readable<User>, Writable<User> { ... }
class AuditLogRepository implements Readable<AuditLog> { ... } // read-only
```

### D — Dependency Inversion
Depend on abstractions. Inject all dependencies through the constructor.

```typescript
// Abstractions
interface Logger {
  info(message: string, meta?: Record<string, unknown>): void;
  error(message: string, error?: Error): void;
}

interface OrderRepository {
  findById(id: string): Promise<Order | null>;
  save(order: Order): Promise<Order>;
}

// High-level class depends only on interfaces
class OrderService {
  constructor(
    private readonly repo: OrderRepository,
    private readonly logger: Logger,
    private readonly events: EventBus,
  ) {}

  async createOrder(dto: CreateOrderDto): Promise<Order> {
    this.logger.info('Creating order', { dto });
    const order = Order.create(dto);
    const saved = await this.repo.save(order);
    this.events.emit('order:created', saved);
    return saved;
  }
}

// Wire up at the composition root — the only place that knows concretions
const orderService = new OrderService(
  new PostgresOrderRepository(db),
  new WinstonLogger(),
  new TypedEventBus(),
);
```

## Design Patterns

### Abstract Base Class — for shared behavior across a family of classes
```typescript
abstract class BaseRepository<T, ID = string> {
  constructor(protected readonly db: Database, protected readonly table: string) {}

  async findById(id: ID): Promise<T | null> {
    const row = await this.db.query(`SELECT * FROM ${this.table} WHERE id = $1`, [id]);
    return row ? this.toDomain(row) : null;
  }

  async save(entity: T): Promise<T> {
    const row = this.toPersistence(entity);
    await this.db.upsert(this.table, row);
    return entity;
  }

  async delete(id: ID): Promise<void> {
    await this.db.query(`DELETE FROM ${this.table} WHERE id = $1`, [id]);
  }

  protected abstract toDomain(row: Record<string, unknown>): T;
  protected abstract toPersistence(entity: T): Record<string, unknown>;
}

class UserRepository extends BaseRepository<User> {
  constructor(db: Database) { super(db, 'users'); }
  protected toDomain(row: Record<string, unknown>): User { ... }
  protected toPersistence(user: User): Record<string, unknown> { ... }
}
```

### Factory — for object creation with complex setup
```typescript
class HttpClientFactory {
  static create(config: ClientConfig): HttpClient {
    const instance = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout ?? 5000,
      headers: { Authorization: `Bearer ${config.token}` },
    });
    return new AxiosHttpClient(instance);
  }
}

class AxiosHttpClient implements HttpClient {
  constructor(private readonly instance: AxiosInstance) {}
  async get<T>(url: string): Promise<T> { return (await this.instance.get<T>(url)).data; }
  async post<T>(url: string, body: unknown): Promise<T> { return (await this.instance.post<T>(url, body)).data; }
}
```

### Strategy — for interchangeable algorithms
```typescript
interface PricingStrategy {
  calculate(basePrice: number, quantity: number): number;
}

class FlatPricing implements PricingStrategy {
  calculate(base: number, qty: number): number { return base * qty; }
}

class TieredPricing implements PricingStrategy {
  constructor(private readonly threshold: number, private readonly discount: number) {}
  calculate(base: number, qty: number): number {
    return qty > this.threshold ? base * qty * (1 - this.discount) : base * qty;
  }
}

class PricingService {
  constructor(private strategy: PricingStrategy) {}
  setStrategy(strategy: PricingStrategy) { this.strategy = strategy; }
  getPrice(base: number, qty: number): number { return this.strategy.calculate(base, qty); }
}
```

### Observer / EventBus — for decoupled communication
```typescript
type EventMap = {
  'order:created': { orderId: string; total: number };
  'order:shipped': { orderId: string; trackingId: string };
};

class TypedEventBus<T extends Record<string, unknown>> {
  private listeners = new Map<keyof T, Set<(data: any) => void>>();

  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(handler);
  }

  off<K extends keyof T>(event: K, handler: (data: T[K]) => void): void {
    this.listeners.get(event)?.delete(handler);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event)?.forEach(fn => fn(data));
  }
}
```

### Singleton — for shared stateful services (use sparingly)
```typescript
class ConfigService {
  private static instance: ConfigService;
  private config: Map<string, unknown> = new Map();

  private constructor() {}

  static getInstance(): ConfigService {
    if (!ConfigService.instance) {
      ConfigService.instance = new ConfigService();
    }
    return ConfigService.instance;
  }

  get<T>(key: string): T | undefined { return this.config.get(key) as T; }
  set(key: string, value: unknown): void { this.config.set(key, value); }
}
```

### Builder — for constructing complex objects step by step
```typescript
class QueryBuilder {
  private table = '';
  private conditions: string[] = [];
  private orderByClause = '';
  private limitValue?: number;

  from(table: string): this { this.table = table; return this; }
  where(condition: string): this { this.conditions.push(condition); return this; }
  orderBy(column: string, dir: 'ASC' | 'DESC' = 'ASC'): this { this.orderByClause = `ORDER BY ${column} ${dir}`; return this; }
  limit(n: number): this { this.limitValue = n; return this; }

  build(): string {
    let sql = `SELECT * FROM ${this.table}`;
    if (this.conditions.length) sql += ` WHERE ${this.conditions.join(' AND ')}`;
    if (this.orderByClause) sql += ` ${this.orderByClause}`;
    if (this.limitValue) sql += ` LIMIT ${this.limitValue}`;
    return sql;
  }
}
```

## Class Design Rules

- **All business logic lives in classes.** Free functions are only for pure utilities (formatDate, slugify, etc.).
- **Constructor injection for all dependencies.** Never instantiate dependencies inside a class.
- **Mark fields `private readonly`** unless mutation is explicitly needed.
- **Use `abstract` classes** for shared behavior across a family. Use interfaces for contracts that cross module boundaries.
- **One class per file.** Name files after the class: `order.service.ts`, `user.repository.ts`.
- **Encapsulate state.** No public fields — use getters/setters when external access is needed.

```typescript
class User {
  private constructor(
    private readonly _id: string,
    private _email: string,
    private _name: string,
  ) {}

  static create(dto: CreateUserDto): User {
    return new User(crypto.randomUUID(), dto.email, dto.name);
  }

  get id(): string { return this._id; }
  get email(): string { return this._email; }
  get name(): string { return this._name; }

  changeName(name: string): void {
    if (!name.trim()) throw new ValidationError('Name cannot be empty');
    this._name = name;
  }
}
```

## TypeScript-Specific Rules

### Types
- Use `interface` for contracts that classes implement. Use `type` for unions, intersections, and mapped types.
- Export types from the module's `types.ts` or alongside the class that defines them.
- Never use `any`. Use `unknown` and narrow with type guards.
- Use branded types for domain identifiers: `type UserId = string & { readonly __brand: 'UserId' }`.

### Error Handling
- Define domain-specific error classes per module, extending a base error.

```typescript
abstract class AppError extends Error {
  abstract readonly code: string;
  abstract readonly statusCode: number;
}

class NotFoundError extends AppError {
  readonly code = 'NOT_FOUND';
  readonly statusCode = 404;
  constructor(entity: string, id: string) {
    super(`${entity} with id '${id}' not found`);
  }
}

class ValidationError extends AppError {
  readonly code = 'VALIDATION_ERROR';
  readonly statusCode = 400;
  constructor(message: string, public readonly fields?: Record<string, string>) {
    super(message);
  }
}
```

### Naming
- `PascalCase` for types, interfaces, classes, enums.
- `camelCase` for variables, methods, properties.
- `UPPER_SNAKE_CASE` for true constants (compile-time values).
- Prefix interfaces with behavior, not `I`: `Readable`, `Cacheable`, not `IReadable`.
- Name files after what they export: `user.repository.ts`, `order.service.ts`.

## Frontend (React / Next.js)

### Component Architecture
- Components are classes or function components — use function components with hooks for UI, but extract all business logic into class-based services/hooks.
- Every component gets its own folder: `ComponentName/ComponentName.tsx`, `index.ts`, test, styles.
- Smart (container) vs dumb (presentational) separation. Containers wire services; presentational components are pure UI.

```typescript
// Presentational — pure props in, JSX out
interface UserCardProps {
  name: string;
  email: string;
  avatar: string;
  onEdit: () => void;
}

function UserCard({ name, email, avatar, onEdit }: UserCardProps) {
  return (
    <div className="user-card">
      <img src={avatar} alt={name} />
      <h3>{name}</h3>
      <p>{email}</p>
      <button onClick={onEdit}>Edit</button>
    </div>
  );
}

// Container — wires services to presentational
function UserCardContainer({ userId }: { userId: string }) {
  const userService = useService(UserService);
  const { data: user, isLoading } = useQuery(['user', userId], () => userService.getById(userId));

  if (isLoading || !user) return <Skeleton />;
  return <UserCard {...user} onEdit={() => userService.openEditModal(userId)} />;
}
```

### Custom Hooks as Service Adapters
Wrap class services in hooks so React components can consume them reactively:

```typescript
class CartService {
  private items: CartItem[] = [];
  
  addItem(product: Product, quantity: number): void { ... }
  removeItem(productId: string): void { ... }
  getTotal(): number { ... }
}

// Hook wraps the service for React consumption
function useCart() {
  const cartService = useService(CartService);
  const [items, setItems] = useState(cartService.getItems());

  const addItem = useCallback((product: Product, qty: number) => {
    cartService.addItem(product, qty);
    setItems([...cartService.getItems()]);
  }, [cartService]);

  return { items, addItem, total: cartService.getTotal() };
}
```

### State Management
- Local state for UI-only concerns (open/closed, form inputs).
- Service classes for business logic state. Expose via hooks.
- Server state via React Query / TanStack Query — never manually cache API responses in useState.
- Global state (auth, theme) via Context + class service, not raw context values.

```typescript
class AuthService {
  private user: User | null = null;

  async login(credentials: LoginDto): Promise<User> {
    const response = await this.api.post<AuthResponse>('/auth/login', credentials);
    this.user = response.user;
    this.tokenStore.set(response.token);
    return this.user;
  }

  getUser(): User | null { return this.user; }
  isAuthenticated(): boolean { return this.user !== null; }
}

// Context provides the service instance
const AuthContext = createContext<AuthService>(null!);
function useAuth() { return useContext(AuthContext); }
```

### Styling
- CSS Modules or Tailwind — no inline styles for anything beyond dynamic values.
- Co-locate styles: `Button.module.css` next to `Button.tsx`.
- Design tokens for colors, spacing, typography — never hardcode values.

### Performance
- `React.memo()` for presentational components that receive stable props.
- `useMemo` / `useCallback` only when profiling shows a real problem — not preemptively.
- Lazy load routes and heavy components with `React.lazy()` / `next/dynamic`.
- Images: use `next/image` or proper `loading="lazy"` attributes.

### Accessibility
- Semantic HTML first: `<button>`, `<nav>`, `<main>`, `<article>` — not `<div onClick>`.
- All interactive elements must be keyboard-accessible.
- Form inputs need associated `<label>` elements.
- ARIA attributes only when semantic HTML is insufficient.
