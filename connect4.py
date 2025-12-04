"""
Hybrid Connect 4 Engine with Neural Network (PyTorch, MPS-aware)
- Integrates a learned neural network with minimax search
- Self-play training capability
- Policy and Value networks in PyTorch
"""

import numpy as np
import copy
from typing import List, Tuple, Dict, Optional
from enum import Enum
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------------------------------------------
# Device selection (Apple MPS if available, else CUDA, else CPU)
# -------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


class Player(Enum):
    """Represents players in the game."""
    AI = 1
    OPPONENT = -1
    EMPTY = 0


class GameState:
    """Manages the board state and game logic."""
    
    ROWS = 6
    COLS = 7
    
    def __init__(self):
        """Initialize an empty board."""
        self.board = [[Player.EMPTY for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.column_heights = [0] * self.COLS
        self.move_history = []
        self.current_player = Player.AI
    
    def get_valid_moves(self) -> List[int]:
        """Return list of valid column indices where a move can be made."""
        return [col for col in range(self.COLS) if self.column_heights[col] < self.ROWS]
    
    def apply_move(self, col: int, player: Player) -> bool:
        """Place a piece in the given column."""
        if col < 0 or col >= self.COLS or self.column_heights[col] >= self.ROWS:
            return False
        
        row = self.ROWS - 1 - self.column_heights[col]
        self.board[row][col] = player
        self.column_heights[col] += 1
        self.move_history.append(col)
        self.current_player = Player.OPPONENT if player == Player.AI else Player.AI
        return True
    
    def undo_move(self) -> bool:
        """Undo the last move."""
        if not self.move_history:
            return False
        
        col = self.move_history.pop()
        self.column_heights[col] -= 1
        row = self.ROWS - 1 - self.column_heights[col]
        self.board[row][col] = Player.EMPTY
        self.current_player = Player.OPPONENT if self.current_player == Player.AI else Player.AI
        return True
    
    def check_win(self, row: int, col: int, player: Player) -> bool:
        """Check if the last move at (row, col) created a 4-in-a-row."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            
            r, c = row + dr, col + dc
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            
            r, c = row - dr, col - dc
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 4:
                return True
        
        return False
    
    def is_game_over(self) -> Tuple[bool, Optional[Player]]:
        """Check if game is over. Returns (is_over, winner)."""
        if self.move_history:
            last_col = self.move_history[-1]
            last_row = self.ROWS - self.column_heights[last_col]
            last_player = self.board[last_row][last_col]
            
            if self.check_win(last_row, last_col, last_player):
                return True, last_player
        
        if len(self.move_history) == self.ROWS * self.COLS:
            return True, Player.EMPTY
        
        return False, None
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        new_state = GameState()
        new_state.board = [row[:] for row in self.board]
        new_state.column_heights = self.column_heights[:]
        new_state.move_history = self.move_history[:]
        new_state.current_player = self.current_player
        return new_state
    
    def get_board_hash(self) -> str:
        """Generate a unique hash for the current board position."""
        return ''.join(
            str(cell.value) for row in self.board for cell in row
        ) + str(self.current_player.value)
    
    def to_tensor(self) -> np.ndarray:
        """
        Convert board to a tensor representation.
        Shape: (1, ROWS, COLS, 2)
        Channel 0: AI pieces
        Channel 1: Opponent pieces
        """
        tensor = np.zeros((1, self.ROWS, self.COLS, 2), dtype=np.float32)
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.board[r][c] == Player.AI:
                    tensor[0, r, c, 0] = 1.0
                elif self.board[r][c] == Player.OPPONENT:
                    tensor[0, r, c, 1] = 1.0
        return tensor
    
    def display(self):
        """Print the board."""
        print("\n  0 1 2 3 4 5 6")
        for row in self.board:
            print(" |" + "|".join(
                " " if cell == Player.EMPTY else ("X" if cell == Player.AI else "O")
                for cell in row
            ) + "|")
        print(" +" + "+".join(["-"] * 7) + "+\n")


# -------------------------------------------------------------------
# PyTorch neural net (policy + value heads)
# -------------------------------------------------------------------
class Connect4Net(nn.Module):
    """
    Simple convolutional neural network for Connect 4
    - Input: (batch, 2, 6, 7)
    - Outputs:
        policy_logits: (batch, 7)
        value: (batch,) in [-1, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 6 * 7, 128)

        # Policy head
        self.policy_head = nn.Linear(128, 7)

        # Value head
        self.value_fc = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 2, 6, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))

        policy_logits = self.policy_head(x)
        v = F.relu(self.value_fc(x))
        v = torch.tanh(self.value_out(v)).squeeze(-1)  # (B,)

        return policy_logits, v


class AlphaZeroConnect4Net(nn.Module):
    """
    AlphaZero-style ResNet for Connect 4
    - Input: (batch, 2, 6, 7)
    - Outputs:
        policy_logits: (batch, 7)
        value: (batch,) in [-1, 1]
    """
    def __init__(self, num_res_blocks: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._build_residual_block(128) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(6 * 7, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def _build_residual_block(self, channels: int) -> nn.Module:
        """Build a single residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x += residual  # Skip connection
            x = F.relu(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


class NeuralNetworkEvaluator:
    """
    Evaluates board positions using a PyTorch neural network.
    Provides both policy (move probabilities) and value (position evaluation).
    """
    
    def __init__(self, model_path: Optional[str] = None, use_alphazero: bool = True):
        """
        Initialize evaluator with optional pre-trained model.
        
        Args:
            model_path: Path to saved model weights (torch .pt or .pth)
            use_alphazero: Use AlphaZeroConnect4Net if True, else Connect4Net
        """
        self.device = DEVICE
        self.model = AlphaZeroConnect4Net().to(self.device) if use_alphazero else Connect4Net().to(self.device)
        self.model.eval()  # default to eval mode

        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _board_to_tensor(self, state: GameState) -> torch.Tensor:
        """
        Convert GameState to a PyTorch tensor on the correct device.
        Input shape (1, 6, 7, 2) -> (1, 2, 6, 7)
        """
        board_np = state.to_tensor()  # (1, 6, 7, 2)
        board_t = torch.from_numpy(board_np).permute(0, 3, 1, 2)  # (1, 2, 6, 7)
        return board_t.to(self.device)
    
    def evaluate(self, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Evaluate a position.
        
        Returns:
            (policy, value_score)
            - policy: numpy array of shape (7,) with probabilities for each move
            - value_score: float in [-1, 1] representing position evaluation
        """
        # Check game over first (hard-coded terminal values)
        is_over, winner = state.is_game_over()
        if is_over:
            if winner == Player.AI:
                value = 1.0
            elif winner == Player.OPPONENT:
                value = -1.0
            else:
                value = 0.0
            policy = np.array([1.0 / 7.0] * 7, dtype=np.float32)
            return policy, value
        
        board_t = self._board_to_tensor(state)

        with torch.no_grad():
            logits, value_t = self.model(board_t)
            policy_t = torch.softmax(logits, dim=-1)  # (1, 7)
        
        policy = policy_t[0].cpu().numpy()
        value = float(value_t[0].cpu().item())

        # Mask invalid moves
        valid_moves = state.get_valid_moves()
        mask = np.zeros(7, dtype=np.float32)
        for move in valid_moves:
            mask[move] = 1.0
        
        policy = policy * mask
        policy = policy / (np.sum(policy) + 1e-8)
        
        return policy, value
    
    def save(self, model_path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), model_path)
    
    def load(self, model_path: str):
        """Load model weights."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model from {model_path}: {e}")


class MCTSNode:
    """Node in the Monte Carlo Tree Search (MCTS)"""
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None):
        """Initialize with a given game state."""
        self.state = state
        self.parent = parent
        self.visits = 0
        self.value = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.untried_actions = state.get_valid_moves()
        self.player = state.current_player
    
    def expand(self, action: int):
        """Expand the node by adding a child for the given action."""
        if action in self.children:
            return self.children[action]
        
        # Apply the action to get the new state
        new_state = self.state.copy()
        new_state.apply_move(action, self.player)
        
        child_node = MCTSNode(new_state, parent=self)
        self.children[action] = child_node
        return child_node
    
    def is_fully_expanded(self) -> bool:
        """Check if the node is fully expanded (all children visited)."""
        return len(self.children) == len(self.untried_actions)


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) for guiding move selection.
    """
    
    def __init__(self, evaluator: NeuralNetworkEvaluator, c_puct: float = 1.0, simulations: int = 100):
        """
        Initialize MCTS.
        
        Args:
            evaluator: NeuralNetworkEvaluator instance for neural network guidance
            c_puct: Exploration parameter for MCTS
            simulations: Number of simulations per move
        """
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.simulations = simulations
    
    def search(self, root: MCTSNode):
        """
        Perform MCTS from the root node.
        
        Args:
            root: MCTSNode representing the current game state
        """
        for _ in range(self.simulations):
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = self._select_child(node)
            
            # Expansion
            if node.untried_actions:
                action = node.untried_actions.pop()
                node = node.expand(action)
            
            # Evaluation (neural network)
            policy, value = self.evaluator.evaluate(node.state)
            
            # Backpropagation
            self._backpropagate(node, value)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select a child node using the UCT (Upper Confidence Bound for Trees) formula.
        """
        log_total_visits = np.log(node.visits + 1e-8)
        def uct_value(child: MCTSNode):
            exploitation = child.value / (child.visits + 1e-8)
            exploration = self.c_puct * np.sqrt(log_total_visits / (child.visits + 1e-8))
            return exploitation + exploration
        
        return max(node.children.values(), key=uct_value)
    
    def _simulate_randomly(self, state: GameState) -> float:
        """
        Perform a random simulation from the given state.
        
        Returns:
            Value in [-1, 1] indicating the outcome from the perspective of the AI
        """
        is_over, winner = state.is_game_over()
        if is_over:
            if winner == Player.AI:
                return 1.0
            elif winner == Player.OPPONENT:
                return -1.0
            else:
                return 0.0
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return 0.0
        
        move = np.random.choice(valid_moves)
        state.apply_move(move, state.current_player)
        return -self._simulate_randomly(state)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate the evaluation result through the visited nodes.
        """
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent


class HybridEngine:
    """
    Hybrid Connect 4 Engine combining neural network with minimax search.
    """
    
    def __init__(self, evaluator: NeuralNetworkEvaluator, max_depth: int = 6):
        """
        Initialize hybrid engine.
        
        Args:
            evaluator: NeuralNetworkEvaluator instance
            max_depth: Maximum search depth
        """
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.transposition_table: Dict[str, Tuple[float, int]] = {}
        self.nodes_evaluated = 0
    
    def find_best_move(self, state: GameState, depth: int = None) -> int:
        """Find the best move using neural-guided search."""
        if depth is None:
            depth = self.max_depth
        
        valid_moves = state.get_valid_moves()
        best_move = valid_moves[0]
        best_value = float('-inf')
        
        # Get neural network's policy to guide move ordering
        policy, _ = self.evaluator.evaluate(state)
        print(f"Sorting through {valid_moves} ")
        for move in tqdm(sorted(valid_moves, key=lambda m: -policy[m])):
            print("applying move!")
            state.apply_move(move, state.current_player)
            value = -self._negamax(state, depth - 1)
            print("undo move!")
            state.undo_move()
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def _negamax(self, state: GameState, depth: int) -> float:
        """
        Negamax search guided by neural network evaluation.
        """
        self.nodes_evaluated += 1
        
        # Terminal node checks
        is_over, winner = state.is_game_over()
        if is_over:
            if winner == Player.AI:
                return 1.0 - (depth / self.max_depth)
            elif winner == Player.OPPONENT:
                return -1.0 + (depth / self.max_depth)
            else:
                return 0.0
        
        # Depth limit: use neural network evaluation
        if depth == 0:
            _, value = self.evaluator.evaluate(state)
            return value
        
        # Transposition table lookup
        board_hash = state.get_board_hash()
        if board_hash in self.transposition_table:
            stored_value, stored_depth = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_value
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return 0.0
        
        # Get policy for move ordering
        policy, _ = self.evaluator.evaluate(state)
        
        best_value = float('-inf')
        for move in sorted(valid_moves, key=lambda m: -policy[m]):
            state.apply_move(move, state.current_player)
            value = -self._negamax(state, depth - 1)
            state.undo_move()
            
            best_value = max(best_value, value)
        
        # Store in transposition table
        self.transposition_table[board_hash] = (best_value, depth)
        
        return best_value
    
    def clear_transposition_table(self):
        """Clear the transposition table."""
        self.transposition_table.clear()
        self.nodes_evaluated = 0


class SelfPlayTrainer:
    """
    Trains the neural network through self-play.
    (Data collection + simple PyTorch training helper)
    """
    
    def __init__(self, evaluator: NeuralNetworkEvaluator, engine: HybridEngine):
        """Initialize trainer."""
        self.evaluator = evaluator
        self.engine = engine
        self.training_data = []
    
    def play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], Player]:
        """
        Play one game of self-play.
        
        Returns:
            (board_states, move_policies, winner)
        """
        state = GameState()
        board_states = []
        move_policies = []
        
        while True:
            is_over, winner = state.is_game_over()
            if is_over:
                break
            
            # Store board state
            board_states.append(state.to_tensor()[0])  # (6, 7, 2)
            
            # Get policy from neural network
            policy, _ = self.evaluator.evaluate(state)
            move_policies.append(policy.astype(np.float32))
            
            # Select move (weighted by policy with some randomness)
            valid_moves = state.get_valid_moves()
            valid_policy = policy.copy()
            invalid_mask = np.ones(7, dtype=np.float32)
            for move in valid_moves:
                invalid_mask[move] = 0
            valid_policy = valid_policy * (1 - invalid_mask)
            valid_policy = valid_policy / (np.sum(valid_policy) + 1e-8)
            
            # Use softmax temperature for exploration
            temperature = 1.0
            probabilities = np.power(valid_policy, 1.0 / temperature)
            probabilities = probabilities / (np.sum(probabilities) + 1e-8)
            
            move = np.random.choice(7, p=probabilities)
            state.apply_move(move, state.current_player)
        
        _, winner = state.is_game_over()
        
        return board_states, move_policies, winner
    
    def train_on_games(self, num_games: int = 10):
        """
        Play multiple games and collect training data.
        
        Args:
            num_games: Number of games to play
        """
        print(f"Playing {num_games} self-play games...")
        
        for game_num in tqdm(range(num_games)):
            board_states, move_policies, winner = self.play_game()
            
            # Compute value targets (from the perspective of AI)
            if winner == Player.AI:
                values = [1.0] * len(board_states)
            elif winner == Player.OPPONENT:
                values = [-1.0] * len(board_states)
            else:
                values = [0.0] * len(board_states)
            
            self.training_data.append({
                'boards': board_states,
                'policies': move_policies,
                'values': values
            })
            
            if (game_num + 1) % 5 == 0:
                print(f"  Completed {game_num + 1}/{num_games} games")
    
    def prepare_training_batch(self) -> Tuple[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """Prepare a full batch for training (all collected data)."""
        if not self.training_data:
            return None, (None, None)
        
        boards = []
        policies = []
        values = []
        
        for game_data in self.training_data:
            boards.extend(game_data['boards'])
            policies.extend(game_data['policies'])
            values.extend(game_data['values'])
        
        boards = np.array(boards, dtype=np.float32)         # (N, 6, 7, 2)
        policies = np.array(policies, dtype=np.float32)     # (N, 7)
        values = np.array(values, dtype=np.float32).reshape(-1, 1)  # (N, 1)
        
        # Shuffle
        indices = np.random.permutation(len(boards))
        boards = boards[indices]
        policies = policies[indices]
        values = values[indices]
        
        return boards, (policies, values)
    
    def clear_training_data(self):
        """Clear training data."""
        self.training_data = []

    def train_network(self, boards: np.ndarray, policies: np.ndarray, values: np.ndarray,
                      epochs: int = 5, batch_size: int = 64, lr: float = 1e-3, 
                      patience: int = 5, min_delta: float = 1e-4):
        """
        Simple PyTorch training loop on collected self-play data with early stopping.
        - boards: (N, 6, 7, 2)
        - policies: (N, 7), probability targets
        - values: (N, 1), scalar in [-1, 1]
        - patience: number of epochs with no improvement after which training stops
        - min_delta: minimum change in loss to qualify as an improvement
        """
        model = self.evaluator.model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Convert to tensors once
        boards_t = torch.from_numpy(boards).permute(0, 3, 1, 2).to(DEVICE)  # (N, 2, 6, 7)
        policies_t = torch.from_numpy(policies).to(DEVICE)                  # (N, 7)
        values_t = torch.from_numpy(values).view(-1).to(DEVICE)             # (N,)

        n_samples = boards_t.size(0)
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                idx = permutation[i:i+batch_size]
                b = boards_t[idx]
                p_target = policies_t[idx]
                v_target = values_t[idx]

                optimizer.zero_grad()
                logits, v_pred = model(b)

                # Policy loss: cross-entropy with soft targets
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(p_target * log_probs).sum(dim=-1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(v_pred, v_target)

                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * b.size(0)

            epoch_loss /= n_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Early stopping check
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        model.eval()


class AlphaZeroTrainer:
    def __init__(self, evaluator: NeuralNetworkEvaluator, mcts: MCTS):
        self.evaluator = evaluator
        self.mcts = mcts
        self.training_data = []
        self.stats = {'ai_wins': 0, 'opponent_wins': 0, 'draws': 0}

    def play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], Player]:
        """Play one game using MCTS-guided moves."""
        state = GameState()
        board_states = []
        move_policies = []

        while True:
            root = MCTSNode(state)
            self.mcts.search(root)

            # Extract policy from visit counts
            if not root.children:
                # No moves explored (shouldn't happen)
                break
                
            valid_moves = sorted(root.children.keys())
            visit_counts = np.array([root.children[move].visits for move in valid_moves])
            policy_full = np.zeros(7, dtype=np.float32)
            for move, count in zip(valid_moves, visit_counts):
                policy_full[move] = count
            policy_full = policy_full / (np.sum(policy_full) + 1e-8)

            # Store training data
            board_states.append(state.to_tensor()[0])  # (6, 7, 2)
            move_policies.append(policy_full)

            # Select move proportional to visit counts
            move = np.random.choice(valid_moves, p=visit_counts / np.sum(visit_counts))
            state.apply_move(move, state.current_player)

            # Check for game over
            is_over, winner = state.is_game_over()
            if is_over:
                break

        return board_states, move_policies, winner
    
    def train_on_games(self, num_games: int = 10):
        """
        Play multiple games and collect training data.
        
        Args:
            num_games: Number of games to play
        """
        print(f"Playing {num_games} self-play games with MCTS...")
        self.stats = {'ai_wins': 0, 'opponent_wins': 0, 'draws': 0}
        
        for game_num in tqdm(range(num_games)):
            board_states, move_policies, winner = self.play_game()
            
            # Track game outcome
            if winner == Player.AI:
                self.stats['ai_wins'] += 1
                values = [1.0] * len(board_states)
            elif winner == Player.OPPONENT:
                self.stats['opponent_wins'] += 1
                values = [-1.0] * len(board_states)
            else:
                self.stats['draws'] += 1
                values = [0.0] * len(board_states)
            
            self.training_data.append({
                'boards': board_states,
                'policies': move_policies,
                'values': values
            })
    
    def prepare_training_batch(self) -> Tuple[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """Prepare a full batch for training (all collected data)."""
        if not self.training_data:
            return None, (None, None)
        
        boards = []
        policies = []
        values = []
        
        for game_data in self.training_data:
            boards.extend(game_data['boards'])
            policies.extend(game_data['policies'])
            values.extend(game_data['values'])
        
        boards = np.array(boards, dtype=np.float32)         # (N, 6, 7, 2)
        policies = np.array(policies, dtype=np.float32)     # (N, 7)
        values = np.array(values, dtype=np.float32).reshape(-1, 1)  # (N, 1)
        
        # Shuffle
        indices = np.random.permutation(len(boards))
        boards = boards[indices]
        policies = policies[indices]
        values = values[indices]
        
        return boards, (policies, values)
    
    def clear_training_data(self):
        """Clear training data."""
        self.training_data = []
    
    def print_stats(self):
        """Print training statistics."""
        total = self.stats['ai_wins'] + self.stats['opponent_wins'] + self.stats['draws']
        if total == 0:
            return
        ai_pct = (self.stats['ai_wins'] / total) * 100
        opp_pct = (self.stats['opponent_wins'] / total) * 100
        draw_pct = (self.stats['draws'] / total) * 100
        print(f"\n  Game Stats: AI Wins: {self.stats['ai_wins']} ({ai_pct:.1f}%) | "
              f"Opponent Wins: {self.stats['opponent_wins']} ({opp_pct:.1f}%) | "
              f"Draws: {self.stats['draws']} ({draw_pct:.1f}%)")
    
    def train_network(self, boards: np.ndarray, policies: np.ndarray, values: np.ndarray,
                      epochs: int = 5, batch_size: int = 64, lr: float = 1e-3, 
                      patience: int = 3, min_delta: float = 1e-4):
        """
        PyTorch training loop on collected self-play data with early stopping.
        """
        model = self.evaluator.model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Convert to tensors once
        boards_t = torch.from_numpy(boards).permute(0, 3, 1, 2).to(DEVICE)  # (N, 2, 6, 7)
        policies_t = torch.from_numpy(policies).to(DEVICE)                  # (N, 7)
        values_t = torch.from_numpy(values).view(-1).to(DEVICE)             # (N,)

        n_samples = boards_t.size(0)
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                idx = permutation[i:i+batch_size]
                b = boards_t[idx]
                p_target = policies_t[idx]
                v_target = values_t[idx]

                optimizer.zero_grad()
                logits, v_pred = model(b)

                # Policy loss: cross-entropy with soft targets
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(p_target * log_probs).sum(dim=-1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(v_pred, v_target)

                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * b.size(0)

            epoch_loss /= n_samples
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Early stopping check
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        model.eval()


class Game:
    """Main game loop."""
    
    def __init__(self, engine_type: str = "hybrid", model_path: Optional[str] = None):
        """
        Initialize a game.
        
        Args:
            engine_type: currently only "hybrid" is supported
            model_path: Path to trained model for hybrid engine
        """
        self.state = GameState()
        self.engine_type = engine_type
        
        if engine_type == "hybrid":
            self.evaluator = NeuralNetworkEvaluator(model_path)
            self.engine = HybridEngine(self.evaluator, max_depth=6)
        else:
            raise ValueError("Only 'hybrid' engine is implemented in this PyTorch version")
    
    def play_human_vs_ai(self):
        """Play an interactive game with human vs AI."""
        print("=== Connect 4: Hybrid Engine (Neural Network + Search, PyTorch) ===")
        print("You are O, AI is X")
        print("Enter column (0-6) to make a move\n")
        
        self.state.display()
        
        while True:
            # Human move
            while True:
                try:
                    col = int(input("Your move (0-6): "))
                    if 0 <= col < self.state.COLS and col in self.state.get_valid_moves():
                        self.state.apply_move(col, Player.OPPONENT)
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number 0-6.")
            
            self.state.display()
            
            is_over, winner = self.state.is_game_over()
            if is_over:
                if winner == Player.OPPONENT:
                    print("You win!")
                elif winner == Player.EMPTY:
                    print("It's a draw!")
                break
            
            # AI move
            print("AI is thinking...")
            best_move = self.engine.find_best_move(self.state)
            self.state.apply_move(best_move, Player.AI)
            print(f"AI plays column {best_move}")
            
            self.state.display()
            
            is_over, winner = self.state.is_game_over()
            if is_over:
                if winner == Player.AI:
                    print("AI wins!")
                elif winner == Player.EMPTY:
                    print("It's a draw!")
                break
            
            stats = {"nodes_evaluated": self.engine.nodes_evaluated}
            print(f"Stats: {stats}")
            self.engine.clear_transposition_table()


if __name__ == "__main__":
    # Example: Play with hybrid engine
    game = Game(engine_type="hybrid")
    game.play_human_vs_ai()
    
    # # AlphaZero training with MCTS
    # print("\n=== Training AlphaZero Connect 4 Engine with MCTS ===")
    # evaluator = NeuralNetworkEvaluator(use_alphazero=True)
    # mcts = MCTS(evaluator, c_puct=1.41, simulations=200)  # Optimized for Mac
    # trainer = AlphaZeroTrainer(evaluator, mcts)

    # num_games_per_iteration = 500
    # num_iterations = 5
    
    # for iteration in range(num_iterations):
    #     print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
    #     trainer.train_on_games(num_games=num_games_per_iteration)
    #     trainer.print_stats()
    #     boards, (policies, values) = trainer.prepare_training_batch()
        
    #     if boards is not None:
    #         print(f"Collected {len(boards)} training examples")
    #         trainer.train_network(boards, policies, values, epochs=5, batch_size=64, lr=1e-3, patience=2)
    #         evaluator.save(f"alphazero_model_iter_{iteration + 1}.pt")
    #     else:
    #         print("No training data collected.")
        
    #     trainer.clear_training_data()
