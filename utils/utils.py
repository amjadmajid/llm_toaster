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
        exit(1)

        

def evaluate_model(model, dataset, criterion, config):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): The validation dataset.
        criterion (nn.Module): The loss function.
        batch_size (int): Batch size for evaluation.
        seq_len (int): Sequence length for evaluation.
        eval_iter (int): Number of evaluation iterations.

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
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def log_training_info(iteration, config, total_loss, interval_start_time, training_start_time):
    current_time = time.time()
    training_time = _format_time(current_time - training_start_time)
    iteration_time = (current_time - interval_start_time) / config.log_inter
    interval_start_time = current_time
    processed_tokens = config.batch_size * config.seq_len * config.n_batches
    tokens_per_sec = processed_tokens // iteration_time
    train_loss = total_loss / config.log_inter
    total_loss = 0

    logger.info(f"Iteration {iteration} | Train Loss {train_loss:.5f} | Training Time: {training_time} | Iteration Time: {iteration_time * 1000:.3f} ms | {tokens_per_sec} tokens/sec")

    return interval_start_time, total_loss