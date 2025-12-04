# Connect 4 AI with AlphaZero and MCTS

A PyTorch-based Connect 4 engine that combines deep learning with Monte Carlo Tree Search (MCTS), inspired by AlphaZero's approach. Train your own AI through self-play and compete against it!

## Features

🧠 **AlphaZero-Style Architecture**
- ResNet-based neural network with 5 residual blocks
- Separate policy and value heads for move selection and position evaluation
- Batch normalization and skip connections for stable training

🔍 **Monte Carlo Tree Search (MCTS)**
- Neural network-guided tree search
- UCT (Upper Confidence Bound for Trees) selection strategy
- Efficient self-play game generation


📊 **Self-Play Training**
- Iterative training loop with multiple iterations
- Win/loss/draw tracking per iteration
- Early stopping with patience mechanism
- Model checkpointing at each iteration


### Setup

```bash
# Clone the repository
git clone https://github.com/pythongiant/Connect4ButAI.git
cd Connect4ButAI

# Install dependencies
pip install torch numpy tqdm

# For Mac with M1/M2 chip, PyTorch should include MPS support automatically
```

## Usage

### Train the AI

```python
from connect4 import NeuralNetworkEvaluator, MCTS, AlphaZeroTrainer

# Initialize evaluator with AlphaZero architecture
evaluator = NeuralNetworkEvaluator(use_alphazero=True)

# Set up MCTS with 200 simulations per move
mcts = MCTS(evaluator, c_puct=1.41, simulations=200)

# Create trainer
trainer = AlphaZeroTrainer(evaluator, mcts)

# Training loop: 2500 games per iteration, 50 iterations total
num_games_per_iteration = 2500
num_iterations = 50

for iteration in range(num_iterations):
    print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
    
    # Play games and collect data
    trainer.train_on_games(num_games=num_games_per_iteration)
    
    # Print statistics (AI wins, opponent wins, draws)
    trainer.print_stats()
    
    # Prepare training batch
    boards, (policies, values) = trainer.prepare_training_batch()
    
    if boards is not None:
        print(f"Collected {len(boards)} training examples")
        
        # Train neural network
        trainer.train_network(boards, policies, values, 
                            epochs=5, batch_size=64, lr=1e-3, patience=2)
        
        # Save checkpoint
        evaluator.save(f"alphazero_model_iter_{iteration + 1}.pt")
    
    # Clear data for next iteration
    trainer.clear_training_data()
```

### Play Against the AI

```python
from connect4 import Game

# Create game with trained model
game = Game(engine_type="hybrid", model_path="alphazero_model_iter_50.pt")

# Play interactively
game.play_human_vs_ai()
```

## Architecture

### Neural Network (AlphaZeroConnect4Net)

```
Input: (batch, 2, 6, 7)  # 2 channels (AI and Opponent pieces)
  ↓
Conv2d(2→128, 3×3) + BatchNorm + ReLU
  ↓
5× Residual Blocks
  ├─ Conv2d(128→128) + BatchNorm + ReLU
  ├─ Conv2d(128→128) + BatchNorm
  └─ Skip Connection + ReLU
  ↓
Policy Head                    Value Head
├─ Conv2d(128→2, 1×1)         ├─ Conv2d(128→1, 1×1)
├─ BatchNorm + ReLU           ├─ BatchNorm + ReLU
├─ Flatten                    ├─ Flatten
├─ Linear(84→7)               ├─ Linear(42→128)
└─ Output: logits             ├─ ReLU
                              ├─ Linear(128→1)
                              └─ Tanh → Output: [-1, 1]
```

### MCTS Algorithm

1. **Selection** - Use UCT formula to traverse tree from root
2. **Expansion** - Add new node for unexplored action
3. **Evaluation** - Use neural network to evaluate position
4. **Backpropagation** - Update visit counts and values up the tree

UCT Formula:
```
UCT(node) = Q(node)/N(node) + c_puct × √(ln(N(parent))/N(node))
```

## Training Details

### Hyperparameters (Mac-Optimized)

- **Games per iteration**: 2500
- **Total iterations**: 50
- **MCTS simulations**: 200 per move
- **Exploration parameter (c_puct)**: 1.41
- **Learning rate**: 1e-3
- **Batch size**: 64
- **Epochs**: 5 per iteration
- **Early stopping patience**: 2 epochs

### Expected Training Time

On Mac M1/M2:
- ~1-2 hours per iteration with 2500 games
- ~50-100 hours total for 50 iterations
- Model saves at each iteration for checkpointing

## Game Statistics Output

After each training iteration, you'll see:

```
Game Stats: AI Wins: 1850 (74.0%) | Opponent Wins: 450 (18.0%) | Draws: 200 (8.0%)
```

This shows how the AI's performance improves over training:
- AI win rate should increase
- Opponent win rate should decrease
- Draw rate indicates complex positions

## Model Files

Models are saved as PyTorch state dictionaries:
- `alphazero_model_iter_1.pt` - After iteration 1
- `alphazero_model_iter_2.pt` - After iteration 2
- ... and so on

Load a model:
```python
evaluator = NeuralNetworkEvaluator(model_path="alphazero_model_iter_50.pt", 
                                   use_alphazero=True)
```

## Device Support

The engine automatically selects the best available device:

1. **Apple MPS** (M1/M2/M3 Macs) - Fastest on Apple Silicon
2. **CUDA** (NVIDIA GPUs) - For systems with NVIDIA GPUs
3. **CPU** - Fallback for other systems

Check which device is being used:
```python
from connect4 import DEVICE
print(f"Using device: {DEVICE}")
```

## Customization

### Adjust Network Architecture

```python
# Increase residual blocks for stronger play (slower training)
evaluator = NeuralNetworkEvaluator(use_alphazero=True)
# Modify AlphaZeroConnect4Net in connect4.py, change num_res_blocks=5 to higher
```

### Tune MCTS Parameters

```python
# More simulations = stronger but slower
mcts = MCTS(evaluator, c_puct=1.41, simulations=400)  # 400 instead of 200

# Different exploration/exploitation balance
mcts = MCTS(evaluator, c_puct=2.0, simulations=200)   # Higher c_puct = more exploration
```

### Adjust Training Parameters

```python
trainer.train_network(boards, policies, values, 
                    epochs=10,        # Train longer per iteration
                    batch_size=32,    # Smaller batches
                    lr=5e-4,          # Lower learning rate
                    patience=5)       # More patience for early stopping
```

## Project Structure

```
connect4/
├── connect4.py           # Main engine implementation
├── alphazero_model_iter_*.pt  # Saved model checkpoints
└── README.md             # This file
```

## Key Classes

- **GameState** - Manages board state and game logic
- **Connect4Net** - Simple baseline CNN architecture
- **AlphaZeroConnect4Net** - ResNet-based architecture (primary)
- **NeuralNetworkEvaluator** - Wraps neural network for board evaluation
- **MCTSNode** - Node in the tree search
- **MCTS** - Monte Carlo Tree Search implementation
- **AlphaZeroTrainer** - Orchestrates self-play and training
- **HybridEngine** - Game engine combining network + minimax search
- **Game** - Interactive game loop

## Performance Notes

- First iteration trains quickly (random play)
- Later iterations take longer as games become strategic
- Win rate plateaus around 70-80% (opponent plays optimally)
- Best results after 30+ iterations of training

## Future Improvements

- [ ] Add opening book
- [ ] Implement endgame tablebase
- [ ] Multi-GPU training support
- [ ] Distributed self-play
- [ ] Web UI for playing against trained model
- [ ] Tournament evaluation system

## References

- AlphaZero: Mastering Chess and Shogi by Self-Play (Silver et al., 2018)
- Monte Carlo Tree Search: A New Framework for Game AI (Browne et al., 2012)
- PyTorch Documentation: https://pytorch.org/

## License

MIT License - Feel free to use and modify for your projects!

## Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Share trained models

## Author

Built with ❤️ for Connect 4 AI enthusiasts

---

**Happy training! 🎮🤖**
