import torch
import torch.nn.functional as F
import logging
import datetime

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

def save_model(model, optimizer, scaler, path):
    """
    Save the model, optimizer, and scaler state to a file.

import datetime Path to the file.
    """
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, path)
    except Exception as e:
        logger.error(f"Error saving model, optimizer, and scaler: {e}")



def load_checkpoint_(model, optimizer, scaler, path, device, inference=False):
    """
    Load the model, optimizer, and scaler state from a file.

    Args:
        model (nn.Module): The model to load into.
        optimizer (Optimizer): The optimizer to load into.
        scaler (GradScaler): The gradient scaler to load into.
        path (str): Path to the file.
        device (str): Device to load the model onto.
    """
    try:
        checkpoint = torch.load(path, map_location=device)
        # Load model state dict
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
        # Load optimizer state dict
        if not inference: 
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Load scaler state dict
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    except Exception as e:
        logger.error(f"Error loading model, optimizer, and scaler: {e}")


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

# def setup_logging(log_file):
#     # Create a custom logger
#     t_logger = logging.getLogger("TRAIN_LOG")
#     t_logger.setLevel(logging.INFO)

#     # Remove all existing handlers from t_logger
#     t_logger.handlers = []
#     t_logger.propagate = False  # Prevent log messages from being passed to ancestor loggers

#     # Create file handler and set level to INFO
#     file_handler = logging.FileHandler(log_file)
#     file_handler.setLevel(logging.INFO)

#     # Create formatter and add it to the handler
#     formatter = logging.Formatter('%(message)s')
#     file_handler.setFormatter(formatter)

#     # Add the handler to t_logger
#     t_logger.addHandler(file_handler)

#     return t_logger


def training_logs(iteration, config, loss, iteration_duration, training_duration):
    # Get current time with milliseconds
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S') + f",{int(now.microsecond / 1000):03d}"

    formatted_td = _format_time(training_duration)
    processed_tokens = config.batch_size * config.seq_len * config.n_batches
    tokens_per_sec = processed_tokens / iteration_duration

    # Format numerical values to match the sample logs
    iteration_time_ms = iteration_duration * 1000  # Convert to milliseconds

    # Create the message
    message = (f"Iteration {iteration} | Train Loss {loss:.5f} | Training Time: {formatted_td} | "
               f"Iteration Time: {iteration_time_ms:.3f} ms | {tokens_per_sec:.1f} tokens/sec")

    # Combine all parts into the final log string
    log_str = f"{timestamp} - {message} \n"

    return log_str

def write_logs(file, logs, append_txt=True):

    with open(file, 'a' if append_txt else 'w') as f:
        f.write(logs)
        f.flush()