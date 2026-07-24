---
title: "Python Basics"
stage: 1
order: 2
minutes: 60
difficulty: beginner
prerequisites: ["how-to-learn-ai"]
tags: ["python", "fundamentals"]
summary: "Variables, lists, dicts, loops, functions, imports, and how to read the error messages you'll see constantly."
---

# Python Basics

## Why this matters

Every lesson after this one assumes you can read and write basic Python.
Not clever Python — just enough to store a value, loop over some data, make a
decision, and package repeated logic into a function. That's the whole toolkit
this lesson builds, and it's the same toolkit you'll still be using when you
write your first neural network many lessons from now.

## The concept

**Variables and types.** A variable is a labelled box that holds a value.
`age = 36` creates a box named `age` and puts `36` in it. Python figures out
the *type* of value from what you put in — you never declare it yourself.

| Type | Example | Meaning |
|---|---|---|
| `int` | `36` | a whole number |
| `float` | `1.68` | a number with a decimal point |
| `str` | `"Ada"` | text, in quotes |
| `bool` | `True` | one of exactly two values: `True` or `False` |

Types matter because operations mean different things for different types:
`"3" + "4"` gives `"34"` (joining text), while `3 + 4` gives `7` (adding
numbers). Mixing them without converting first (`int("3") + 4`) is one of the
most common beginner errors.

**Lists and dicts.** A list is an ordered sequence you access by position:
`languages[0]` is the first item. A dict (dictionary) is a lookup table you
access by key instead of position: `profile["name"]` gets whatever value was
stored under the key `"name"`, regardless of where it sits. Reach for a list
when order matters and items are similar ("all my scores"); reach for a dict
when you're labelling different pieces of one thing ("a person's name, age,
and city").

**`for` and `if`.** A `for` loop repeats a block of code once per item in a
collection: `for lang in languages:` runs its indented body once for each
language, binding `lang` to the current one each time. An `if` statement
branches: it runs its indented body only when the condition is `True`, and an
optional `else` block runs otherwise. Indentation is not just style in
Python — it's how the language knows which lines belong to the loop or the
branch. Four spaces per level is the standard, and mixing tabs and spaces will
break your code.

**Functions.** A function packages a piece of logic under a name so you can
reuse it without retyping it. `def greet(person_name):` defines a function
called `greet` that takes one input (a *parameter* named `person_name`); the
indented body runs each time you *call* it, like `greet("Ada")`. `return`
sends a value back out to whoever called the function; `print` just displays
something and sends nothing back — they're easy to confuse when you're
starting out, but only `return` lets you use the result afterwards.

**Imports.** Nobody writes everything from scratch. `import math` loads
Python's built-in math toolkit into your program so you can use things like
`math.sqrt(9)`. Later lessons will `import numpy` and `import pandas` the same
way — those aren't part of core Python, which is what virtual environments
are for.

**Virtual environments.** A virtual environment is an isolated, project-local
folder of installed packages, so that project A can use `pandas` version 1
while project B uses version 2 without them colliding. You create one with
`python -m venv .venv`, then activate it (`source .venv/bin/activate` on
macOS/Linux, `.venv\Scripts\activate` on Windows) before installing anything
with `pip install`. Every serious Python project you touch from here on should
have its own virtual environment — it's the difference between "it works on
my machine" and packages fighting each other across every project you have.

**Reading a traceback.** When Python hits an error it can't recover from, it
prints a *traceback* instead of your program's normal output. Read it from
the **bottom up**: the last line names the error type and what went wrong
(`IndexError: list index out of range`); the lines above it show the exact
line of code that caused it, and above that, what called that code, and so
on. Beginners often panic and reread from the top — train yourself to read
the last line first, then walk upward only if you need more context.

## In code

```python
name = "Ada"
age = 36
height = 1.68
is_learning = True

print(type(name), type(age), type(height), type(is_learning))

languages = ["python", "sql", "python"]

profile = {"name": name, "age": age, "languages": languages}

for lang in languages:
    if lang == "python":
        print(lang, "is great for beginners")
    else:
        print(lang, "is also useful")


def greet(person_name):
    return f"Hello, {person_name}!"


print(greet(profile["name"]))

import math

print(round(math.sqrt(profile["age"]), 2))
```

```
<class 'str'> <class 'int'> <class 'float'> <class 'bool'>
python is great for beginners
sql is also useful
python is great for beginners
Hello, Ada!
6.0
```

Now the same list, but read past its end — this is the traceback you'll learn
to read:

```python
prices = [10, 20, 30]
print(prices[5])
```

```
Traceback (most recent call last):
  File "script.py", line 2, in <module>
    print(prices[5])
IndexError: list index out of range
```

Bottom line first: `IndexError: list index out of range` — you asked for a
position that doesn't exist in a 3-item list. The line above shows exactly
which line did it. That's the whole skill.

## Build this

Write a script that stores a list of numbers, then reports the **mean**,
**min**, and **max** — without using any library (no `import`, no
`sum()`/`min()`/`max()` built-ins either; use a `for` loop to compute all
three yourself). Start from a list like:

```python
numbers = [4, 8, 15, 16, 23, 42]
```

Your script should loop once over `numbers`, tracking a running total plus the
smallest and largest value seen so far, and print all three results at the
end.

**Stretch:** wrap your logic in a function `summarize(numbers)` that returns
the three values, call it on two or three different lists, and add an `if`
that prints `"empty list"` instead of crashing when you pass it `[]`. Then try
running it on `[]` without that check first, so you see the traceback it
would otherwise produce.

## Go deeper

- [Python's official tutorial](https://docs.python.org/3/tutorial/introduction.html) — the primary source, written by the language's own maintainers.
- [freeCodeCamp: The Python Handbook](https://www.freecodecamp.org/news/the-python-handbook/) — a longer walkthrough covering data types, control flow, functions, and virtual environments.
- [Real Python: Python Data Types](https://realpython.com/python-data-types/) — a deeper look at the type table above.
- [Real Python: Understanding the Python Traceback](https://realpython.com/python-traceback/) — worth reading in full once tracebacks stop feeling scary and start feeling useful.
- [Python venv documentation](https://docs.python.org/3/library/venv.html) — the official reference for virtual environments.

**Next:** [NumPy](03-numpy.md)
