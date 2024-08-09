import torch
import torch.nn.functional as F
import logging
import time

logger = logging.getLogger(__name__)

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    Args:
        model (nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return f"{round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000_000, 2)}M"

def save_model(model, path):
    """
    Save the model to a file.

    Args:
        model (nn.Module): The model to save.
        path (str): Path to the file.
    """
    try:
        torch.save(model.state_dict(), path)
        # logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def load_model_weights_(model, path, device):
    """
    Load the model from a file.

    Args:
        model (nn.Module): The model to load into.
        path (str): Path to the file.
        device (str): Device to load the model onto.
    """
    # try:
    #     model.load_state_dict(torch.load(path, map_location=device))
    #     logger.info(f"Model loaded from {path}")
    # except Exception as e:
    #     logger.error(f"Error loading model: {e}")

        # Load the model state dictionary
    try:
        # Load the saved state dictionary
        
        state_dict = torch.load(path , map_location=device)

        # Remove the "_orig_mod." prefix if it exists
        new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(new_state_dict)

    except Exception as e:
        logger.error(f"Error loading model: {e}")

def evaluate_model(model, dataset, criterion, config):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): The validation dataset.
        criterion (nn.Module): The loss function.
        config (ConfigHandler): The configurations

    Returns:
        float: Average validation loss.
    """
    model.eval()
    val_loss = 0
    for _ in range(config.eval_iter):
        # X, Y = dataset.get_rand_batch(batch_size, seq_len)
        X, Y, _ = dataset.next_batch()
        X = torch.tensor(X, dtype=torch.long).to(config.device)
        Y = torch.tensor(Y, dtype=torch.long).to(config.device)
        with torch.no_grad():
            logits = model(X)
            loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
            val_loss += loss.item()
    model.train()
    return val_loss / config.eval_iter


def _format_time(seconds):
    """Format time in seconds to hours, minutes, and seconds."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):04d}h {int(minutes):02d}m {int(seconds):02d}s"

def setup_logging(log_file):
    # Create a custom logger
    t_logger = logging.getLogger("TRAIN_LOG")
    t_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    
    # Optional: Add rotating file handler for log rotation
    # from logging.handlers import RotatingFileHandler
    # file_handler = RotatingFileHandler(log_file, maxBytes=2000, backupCount=5)

    formatter_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter_file)

    t_logger.addHandler(file_handler)

    return t_logger

def log_training_info(t_logger, iteration, config, total_loss, iteration_duration, training_duration):
    formatted_td = _format_time(training_duration)

    processed_tokens = config.batch_size * config.seq_len * config.n_batches
    tokens_per_sec = processed_tokens // iteration_duration
    train_loss = total_loss / config.log_inter

    # this will log to a file and the terminal at the same time
    t_logger.info(f"Iteration {iteration} | Train Loss {train_loss:.5f} | Training Time: {formatted_td} | Iteration Time: {iteration_duration * 1000:.3f} ms | {tokens_per_sec} tokens/sec")