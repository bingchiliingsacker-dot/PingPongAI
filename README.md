# PingPongAI
The first AI project I created with the help of some AI's (I hope to remove this dependency in the future) where 2 different AI's with different mindsets compete with each other inside a pygame simulation window. (Desktop Only)

# Structure
- PingPongAI needs a specific structure to run:

```
PingPongAI
|____ src/
|        |
|        |___ lib.rs
|        |___ Environment.rs # Environment
|
|____ src_py/
|        |
|        |___ __init__.py
|        |___ Trainer.py # Run trainer loop here, takes about 10 minutes
|        |___ AI.py # Neural Network
|
|____ main.py # Run the games here here
|
|____ Cargo.toml
|
|____ Cargo.lock
```

- Note: If you wish to rename some files/directories/functions, you must also change the lines that uses them.

# Pre-Made Models
- If you do not want to sit around for 10 minutes to around 1 hour, PingPongAI comes with a premade model for Blue and Red
- Fun Fact: Blue is the smarter one of the two
