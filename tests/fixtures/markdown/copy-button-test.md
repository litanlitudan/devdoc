# Copy Button Test

This file tests the copy-to-clipboard functionality for code blocks.

## Regular Code Block

```javascript
function greet(name) {
  console.log(`Hello, ${name}!`);
  return `Welcome, ${name}`;
}

greet('World');
```

## Python Example

```python
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    result = a + b
    return result

print(calculate_sum(5, 10))
```

## Inline Code

This is `inline code` that should not have a copy button.

## Multiple Code Blocks

```bash
npm install markserv
npm test
```

```json
{
  "name": "example",
  "version": "1.0.0"
}
```

## Large Code Block with Line Numbers

This should show line numbers with a copy button:

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
}

class UserService {
  private users: User[] = [];

  addUser(user: User): void {
    this.users.push(user);
  }

  getUserById(id: number): User | undefined {
    return this.users.find(u => u.id === id);
  }

  getAllUsers(): User[] {
    return this.users;
  }
}

const service = new UserService();
service.addUser({
  id: 1,
  name: 'John Doe',
  email: 'john@example.com',
  role: 'admin'
});
```
