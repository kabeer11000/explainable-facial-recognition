# Xplainable Facial Recognition.
![image](https://github.com/user-attachments/assets/1421eaa3-41bd-4b9a-a3c4-5999f89788df)

Folder Structure:
`web`: Contains website / client side code.
`recogniser`: Contains code for the face classifier (Identification Provider / Bigger Model).
`face-detector`: Written in CPP / JS compiled to web assembly etc provides bounding box creation and other functionality.
`documentation`: Contains project reports and other related documents.
`colab`: Colab notebooks and more.


# Coding Conventions
1. General Principles
The following principles apply universally to both C++ and TypeScript code within this project:

Readability First: Code must be clear, concise, and easy to understand by any team member. Prioritize clarity over cleverness.

Maintainability: Write code that is straightforward to debug, refactor, and extend.

Algorithmic Efficiency: Be mindful of the time and space complexity of algorithms. Choose appropriate data structures and algorithms to ensure optimal performance. If an algorithm's complexity is not immediately obvious, or if a particular implementation choice has significant performance implications, add comments to explain the rationale.

Minimal Dependencies: For core functionality, strive to use the least number of external libraries possible. This reduces project overhead, potential security vulnerabilities, and simplifies build processes.

Avoid Global Namespace Pollution: Limit the introduction of global variables or functions. Utilize language-specific mechanisms (namespaces in C++, modules in TypeScript) to encapsulate code and prevent naming conflicts.

Immutability (Where Practical): Favor immutable data structures and const declarations. Minimize direct mutation of variables and objects, especially for shared data, to reduce side effects and simplify reasoning about code flow.

No Single-Letter Variables: Avoid single-letter variable names unless they are standard loop counters (e.g., i, j) in very short, localized loops.

Descriptive Naming with Comments: Variable and function names should be descriptive. If a name becomes excessively long or its meaning remains ambiguous despite being descriptive, add concise comments to clarify its purpose or the logic it represents.

2. C++ Conventions
2.1 Naming Conventions
Variables (local, member): camelCase (e.g., totalCount, userName).

Functions/Methods: camelCase (e.g., calculateSum, processData).

Classes/Structs: PascalCase (e.g., UserProfile, HttpRequest).

Constants (global, static const members): SCREAMING_SNAKE_CASE (e.g., MAX_BUFFER_SIZE, DEFAULT_TIMEOUT_MS).

Enums/Enum Classes: PascalCase for the enum type, PascalCase for enumerators (e.g., enum class Status { Pending, Completed };).

Namespaces: lowercase (e.g., project::core).

2.2 Memory Management
RAII (Resource Acquisition Is Initialization): Always prefer RAII principles. Use smart pointers (std::unique_ptr, std::shared_ptr) to manage dynamically allocated memory.

Explicit Deallocation: If raw pointers and new are absolutely necessary (e.g., for C-style APIs or specific performance optimizations), ensure that delete is called exactly once for every new. This should be an exception, not the norm, and must be clearly documented.

No malloc/free: Do not use C-style memory allocation functions unless interacting with specific C libraries that require them.

2.3 Control Flow
Minimize else: Structure code to reduce the need for else blocks. Use early returns (guard clauses) to handle error conditions or base cases at the beginning of functions.

Preferred:

void process(int value) {
    if (value < 0) {
        return; // Handle invalid input
    }
    // Continue with positive value processing
}

Avoided:

void process(int value) {
    if (value >= 0) {
        // Process positive value
    } else {
        // Handle invalid input
    }
}

Nesting Limit: Limit code block nesting to a maximum of two levels. If more nesting is required, refactor the inner logic into a separate function.

Brace Removal (Conditional): For single-statement if, for, while bodies, braces may be omitted for brevity, provided it enhances readability and the single statement is on the same line or immediately following.

Allowed:

if (condition) return;
if (anotherCondition)
    doSomething();

Preferred (for clarity in most cases):

if (condition) {
    return;
}

2.4 Typing and Constants
Proper Typing: Use explicit and appropriate C++ types for all variables, function parameters, and return values. Avoid auto where the type is not immediately obvious from the initializer.

const Usage: Use const extensively for variables, function parameters (pass by const reference), and member functions that do not modify the object's state. This improves safety and allows for compiler optimizations.

No var: The var keyword is not applicable in C++.

2.5 Concurrency
std::thread and std::async: Use std::thread for explicit thread management and std::async for launching asynchronous tasks that return a future.

Synchronization Primitives: When sharing data between threads, use std::mutex, std::unique_lock, std::shared_mutex, and std::condition_variable to prevent race conditions. Avoid raw locks where possible; prefer RAII-style locks.

Thread Safety: Design classes and functions to be thread-safe when they are intended for concurrent access.

3. TypeScript Conventions
3.1 Naming Conventions
Variables (local, member): camelCase (e.g., userCount, isActive).

Functions/Methods: camelCase (e.g., fetchData, updateStatus).

Classes/Interfaces/Types: PascalCase (e.g., UserInterface, ProductType).

Constants (global, module-level): SCREAMING_SNAKE_CASE (e.g., API_BASE_URL, DEFAULT_PAGE_SIZE).

Enums: PascalCase for the enum type, PascalCase for enumerators (e.g., enum UserRole { Admin, Editor, Viewer };).

3.2 Typing
Strong Typing: Always use explicit TypeScript types for variables, function parameters, and return values. Avoid any unless absolutely necessary (e.g., when dealing with truly dynamic external data) and document its usage.

Interfaces and Types: Define interfaces (interface) or type aliases (type) for complex object shapes and function signatures to improve code clarity and maintainability.

3.3 Control Flow
Minimize else: Similar to C++, prioritize early returns (guard clauses) to handle conditional logic and reduce else blocks.

Preferred:

function processInput(input: string): boolean {
    if (!input) {
        return false; // Handle empty input
    }
    // Continue processing valid input
    return true;
}

Nesting Limit: Limit code block nesting to a maximum of two levels. Refactor complex nested logic into separate functions.

Brace Removal (Conditional): For single-statement if, for, while bodies, braces may be omitted for brevity, provided it enhances readability and the single statement is on the same line or immediately following.

Allowed:

if (isValid) return;
for (let i = 0; i < items.length; i++)
    console.log(items[i]);

Preferred (for clarity in most cases):

if (isValid) {
    return;
}

3.4 Variable Declaration and Immutability
const and let: Use const for variables that will not be reassigned. Use let for variables that will be reassigned.

No var: The var keyword is strictly prohibited.

Immutability: Favor immutable operations for arrays and objects. Instead of modifying an array in place, use methods like map(), filter(), reduce(), or the spread operator (...) to create new arrays. For objects, use the spread operator to create new objects with updated properties.

Preferred (Array): const newArray = [...originalArray, newItem];

Preferred (Object): const newObject = { ...originalObject, updatedProperty: newValue };

3.5 Functions
Arrow Functions: Use arrow functions (=>) where possible, especially for callbacks, to maintain lexical this binding and for concise syntax.

Preferred:

const multiply = (a: number, b: number) => a * b;
myArray.map(item => item.id);

Function Signature: Clearly define parameter types and return types for all functions.

3.6 Asynchronous Code
async/await: Use async/await for handling Promises. This makes asynchronous code appear more synchronous and readable.

Error Handling: Always include try...catch blocks with async/await to handle potential errors in asynchronous operations.

Concurrency (JavaScript's Nature): While JavaScript is single-threaded, use async/await to manage non-blocking operations effectively. For true parallel computation (e.g., heavy calculations), consider using Web Workers, but ensure their usage is justified and well-documented.

3.7 Modules
ES Modules: Use import and export statements to manage dependencies and organize code into modules. This prevents global namespace pollution and promotes code reusability.

Default Exports: Limit default exports to one per file, typically the main entity defined in that file. Prefer named exports for all other components.
