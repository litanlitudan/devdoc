// Test JavaScript file for syntax highlighting
const greeting = "Hello, World!";

function sayHello(name) {
    return `${greeting} My name is ${name}`;
}

class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    greet() {
        return sayHello(this.name);
    }
}

// Export the class
module.exports = Person;