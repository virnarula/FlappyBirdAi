# FlappyBirdAi
Training an AI to play Flappy Bird on Pygame. Saw it the internet. Thought it was cool. Made my own


## Play the Game 
```
python game.py
```

## Train a Model
```
python train.py
```

## Test a pre-tained model on a game
```
python game.py --model model.pth
```


## Run with the analytical SimpleModel and debug logging
- Simple heuristic model:
```
python game.py --model_type SimpleModel
```

- Enable concise debug logs to stdout (logs jumps, points, crashes, optional periodic state):
```
python game.py --model_type SimpleModel --debug --debug_state_interval 30
```

- Write debug logs to a file instead of stdout:
```
python game.py --model_type SimpleModel --debug --debug_file flappy_debug.log --debug_state_interval 30
```

Flags:
- `--debug`: Enable compact event logs (JUMP/POINT/CRASH). Low volume by default.
- `--debug_file`: Path to write logs. If omitted, logs go to stdout.
- `--debug_state_interval N`: Also log one compact STATE line every N frames. Use a moderate N (e.g., 30–120) to avoid excessive output.
