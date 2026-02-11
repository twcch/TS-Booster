"""
P6: Adaptive Multi-Objective Loss Weighting
P6.1: Added smoothing mechanism to prevent weight oscillation

Dynamically adjusts loss weights based on base and expert performance.
Uses EMA smoothing on gaps and momentum on weight updates for stability.
"""
import torch
import torch.nn.functional as F


class AdaptiveWeightScheduler:
    """
    Monitors base and expert performance, dynamically adjusts loss weights
    to achieve balanced multi-objective optimization.
    """

    def __init__(self, warmup_epochs, verbose=True):
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        # Performance tracking
        self.history = {
            'base': [],
            'expert': [],
            'overall': []
        }

        self.best_base_loss = float('inf')
        self.best_expert_loss = float('inf')
        self.best_overall_loss = float('inf')

        # Weight history for analysis
        self.weight_history = {
            'aux': [],
            'base': []
        }

        # P6.1: Smoothing mechanism to prevent oscillation
        self.smoothed_base_gap = None
        self.smoothed_expert_gap = None
        self.prev_aux_weight = 0.3  # Initial aux weight for search phase
        self.prev_base_weight = 0.1  # Initial base weight for search phase

    def update(self, base_loss, expert_loss, overall_loss, epoch):
        """Update performance history"""
        self.history['base'].append(base_loss)
        self.history['expert'].append(expert_loss)
        self.history['overall'].append(overall_loss)

        # Track best
        improved = False
        if base_loss < self.best_base_loss:
            self.best_base_loss = base_loss
            improved = True
        if expert_loss < self.best_expert_loss:
            self.best_expert_loss = expert_loss
            improved = True
        if overall_loss < self.best_overall_loss:
            self.best_overall_loss = overall_loss
            improved = True

        if self.verbose and improved and epoch > self.warmup_epochs:
            print(f"    [P6 Adaptive] Best losses - Base: {self.best_base_loss:.4f}, Expert: {self.best_expert_loss:.4f}, Overall: {self.best_overall_loss:.4f}")

    def get_weights(self, epoch):
        """
        Calculate adaptive weights based on current performance

        Strategy:
        - Warmup: Focus on overall learning (low aux/base weights)
        - Search: Dynamically adjust based on performance gaps
          - If base lags behind: increase base_weight
          - If expert lags behind: increase aux_weight
          - If balanced: moderate weights

        P6.1 Enhancement: Smoothing mechanism
        - EMA smoothing on gaps to reduce noise
        - Momentum on weight updates to prevent oscillation
        """
        if epoch < self.warmup_epochs:
            # Warmup phase: minimal auxiliary losses
            weights = {'aux': 0.1, 'base': 0.0}
            self.weight_history['aux'].append(0.1)
            self.weight_history['base'].append(0.0)
            self.prev_aux_weight = 0.1
            self.prev_base_weight = 0.0
            return weights

        # Need at least one epoch of history
        if len(self.history['base']) < 2:
            weights = {'aux': 0.3, 'base': 0.1}
            self.weight_history['aux'].append(0.3)
            self.weight_history['base'].append(0.1)
            self.prev_aux_weight = 0.3
            self.prev_base_weight = 0.1
            return weights

        # Calculate raw performance gaps (current - best)
        raw_base_gap = self.history['base'][-1] - self.best_base_loss
        raw_expert_gap = self.history['expert'][-1] - self.best_expert_loss

        # P6.1: EMA smoothing on gaps (alpha=0.7 → keep 70% old, add 30% new)
        if self.smoothed_base_gap is None:
            # First search epoch: initialize with raw gaps
            self.smoothed_base_gap = raw_base_gap
            self.smoothed_expert_gap = raw_expert_gap
        else:
            # Apply EMA smoothing
            ema_alpha = 0.7
            self.smoothed_base_gap = ema_alpha * self.smoothed_base_gap + (1 - ema_alpha) * raw_base_gap
            self.smoothed_expert_gap = ema_alpha * self.smoothed_expert_gap + (1 - ema_alpha) * raw_expert_gap

        # Calculate relative gaps from smoothed values
        total_gap = self.smoothed_base_gap + self.smoothed_expert_gap + 1e-6
        base_gap_ratio = self.smoothed_base_gap / total_gap
        expert_gap_ratio = self.smoothed_expert_gap / total_gap

        # Calculate target weights based on smoothed gaps
        target_base_weight = 0.05 + 0.35 * base_gap_ratio
        target_aux_weight = 0.2 + 0.6 * expert_gap_ratio

        # Clamp targets to reasonable ranges
        target_base_weight = max(0.05, min(0.4, target_base_weight))
        target_aux_weight = max(0.2, min(0.8, target_aux_weight))

        # P6.1: Momentum smoothing (beta=0.7 → keep 70% old weight, move 30% toward target)
        momentum_beta = 0.7
        aux_weight = momentum_beta * self.prev_aux_weight + (1 - momentum_beta) * target_aux_weight
        base_weight = momentum_beta * self.prev_base_weight + (1 - momentum_beta) * target_base_weight

        # Store current weights for next iteration
        self.prev_aux_weight = aux_weight
        self.prev_base_weight = base_weight

        weights = {'aux': aux_weight, 'base': base_weight}
        self.weight_history['aux'].append(aux_weight)
        self.weight_history['base'].append(base_weight)

        # Log weight changes (only on significant changes)
        if len(self.weight_history['aux']) > 1:
            prev_aux = self.weight_history['aux'][-2]
            if abs(aux_weight - prev_aux) > 0.05 and self.verbose:
                print(f"    [P6.1 Smoothed] Weight adjustment - Aux: {prev_aux:.2f}→{aux_weight:.2f}, Base: {base_weight:.2f}")
                print(f"                    Raw gaps - Base: {raw_base_gap:.4f}, Expert: {raw_expert_gap:.4f}")
                print(f"                    Smoothed gaps - Base: {self.smoothed_base_gap:.4f}, Expert: {self.smoothed_expert_gap:.4f}")

        return weights

    def get_summary(self):
        """Return summary statistics for analysis"""
        return {
            'best_base': self.best_base_loss,
            'best_expert': self.best_expert_loss,
            'best_overall': self.best_overall_loss,
            'final_aux_weight': self.weight_history['aux'][-1] if self.weight_history['aux'] else 0.0,
            'final_base_weight': self.weight_history['base'][-1] if self.weight_history['base'] else 0.0,
            'avg_aux_weight': sum(self.weight_history['aux']) / len(self.weight_history['aux']) if self.weight_history['aux'] else 0.0,
            'avg_base_weight': sum(self.weight_history['base']) / len(self.weight_history['base']) if self.weight_history['base'] else 0.0,
            'final_smoothed_base_gap': self.smoothed_base_gap if self.smoothed_base_gap is not None else 0.0,
            'final_smoothed_expert_gap': self.smoothed_expert_gap if self.smoothed_expert_gap is not None else 0.0,
        }
