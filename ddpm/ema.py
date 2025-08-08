"""
Exponential Moving Average (EMA) Implementation

This module implements EMA for model parameters, which helps stabilize training
and often improves the quality of generated samples in diffusion models.
"""

class EMA():
    """
    Exponential Moving Average for model parameters.
    
    EMA maintains a running average of model parameters that changes more slowly
    than the actual training parameters. This often leads to better and more
    stable results during inference.
    
    The EMA update rule is: ema_param = decay * ema_param + (1 - decay) * current_param
    """
    
    def __init__(self, decay):
        """
        Initialize EMA with given decay rate.
        
        Args:
            decay: Decay rate for EMA (typically close to 1.0, e.g., 0.9999)
                  Higher values mean slower updates to the EMA parameters
        """
        self.decay = decay
    
    def update_average(self, old, new):
        """
        Update a single EMA value with a new value.
        
        Args:
            old: Previous EMA value (or None for first update)
            new: New value to incorporate into the average
            
        Returns:
            Updated EMA value
        """
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        """
        Update all EMA model parameters with current model parameters.
        
        This iterates through all parameters of both models and applies
        the EMA update rule to each parameter tensor.
        
        Args:
            ema_model: Model holding the EMA parameters (gets updated in-place)
            current_model: Current training model with latest parameters
        """
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)