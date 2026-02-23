# Chess-Game

This repository includes two playable 2D chess games in Python:

- `chess_game.py` (Pygame version)
- `chess_game_314.py` (Tkinter version, Python 3.14-friendly)

## Run (Python 3.14 compatible path)

```bash
python -m pip install --upgrade pip
python chess_game_314.py
```

No external dependency is required for `chess_game_314.py` (Tkinter ships with standard Python installers).

## Run (Pygame version)

```bash
python -m pip install pygame
python chess_game.py
```

## Features

- Click and drag pieces to legal target squares.
- Play against a computer opponent.
- Difficulty selector: Easy / Medium / Hard.
- New game button and side switching.
- Check/checkmate/stalemate detection.
