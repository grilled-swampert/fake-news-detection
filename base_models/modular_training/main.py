import importlib
import sys
import os
import torch
import multiprocessing
import datetime  # <-- Added
from training_pipeline import run_training # Our main pipeline function

# --- NEW: Class to log all terminal output ---
class TerminalLogger:
    """
    This class redirects 'print' statements (stdout)
    to both the terminal and a log file.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout # Save the original stdout
        
        # Ensure log directory exists
        log_dir = os.path.dirname(filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        # This flush method is needed for compatibility
        self.terminal.flush()
        self.log_file.flush()
    
    def __del__(self):
        # Restore original stdout and close file on exit
        if hasattr(self, 'terminal'):
             sys.stdout = self.terminal
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

# --- End of new class ---


# --- Hardware & Resource Configuration (Shared) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Note: The print statements below will be captured by the logger
# *after* it is initialized in __main__

# --- Model Registry ---
# We use a list to give them numbers
MODEL_REGISTRY = [
    "muril",
    "indicbert",
    "xlm-roberta",
    "malayalam-bert",
    "llama2",
    "openhathi",
    "sarvam-1",
]

def get_user_choice():
    """
    Displays a menu and asks the user to choose a model.
    Returns the chosen model_name (str) or None if they quit.
    """
    print("\n--- 🤖 Model Training Menu ---")
    print("Please choose a model to train:")
    
    # 1. Print all available models with a number
    for i, model_name in enumerate(MODEL_REGISTRY, 1):
        print(f"  [{i}] {model_name}")
    
    print("  [q] Quit")
    print("---------------------------------")

    while True:
        try:
            choice = input(f"Enter your choice (1-{len(MODEL_REGISTRY)}) or 'q' to quit: ")
            
            if choice.lower() == 'q':
                return None # Signal to quit

            # 2. Check if the choice is a valid number
            choice_num = int(choice)
            if 1 <= choice_num <= len(MODEL_REGISTRY):
                # 3. Get the model name from the list
                selected_model = MODEL_REGISTRY[choice_num - 1]
                return selected_model
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(MODEL_REGISTRY)}.")

        except ValueError:
            print(f"Invalid input. Please enter a number (1-{len(MODEL_REGISTRY)}) or 'q'.")

if __name__ == "__main__":
    # Ensure safe multiprocessing on Windows
    multiprocessing.set_start_method("spawn", force=True)

    # --- NEW: Setup Terminal Logging ---
    # Create a unique log file name with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    terminal_log_path = f"logs/full_terminal_log_{timestamp}.txt"
    
    # We print this *before* redirecting, so the user sees it
    # on the console no matter what.
    print(f"Full terminal output will be saved to: {terminal_log_path}")
    
    # Redirect stdout to our new logger class
    sys.stdout = TerminalLogger(terminal_log_path)
    
    # Redirect stderr (errors) to the same place
    sys.stderr = sys.stdout 
    # --- End of new setup ---

    # --- These print statements will now be logged ---
    print(f"Using {os.cpu_count()} total CPU threads.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
    # ---

    # 1. Ask for the CSV path ONCE at the beginning
    print("\n--- 📄 Data Setup ---")
    default_path = r"D:\fake-news-detection\dataset\fr_dataset_final_balanced.csv"
    csv_path = input(f"Enter the path to your CSV file [press Enter for default]:\n(Default: {default_path})\n")
    
    if not csv_path:
        csv_path = default_path
        
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'. Please check the path and try again.")
        sys.exit(1)
        
    print(f"Using data from: {csv_path}\n")


    # 2. Start the selection loop
    while True:
        model_name = get_user_choice()

        if model_name is None:
            print("Exiting program. Goodbye! 👋")
            break # Exit the while loop and end the script

        # 3. Map the chosen name to the module path
        # Replaces '-' with '_' for correct Python module naming (e.g., "malayalam-bert" -> "malayalam_bert")
        module_name = model_name.replace('-', '_')
        module_path = f"models.{module_name}" 

        # 4. Dynamically import the correct model module
        try:
            print(f"Importing module: {module_path}")
            model_module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"Error: Could not import module '{module_path}'.")
            print("Did you forget to create the file 'models/{module_name}.py'?")
            print(f"Details: {e}")
            sys.exit(1)
        
        # 5. Run the main training pipeline
        run_training(
            model_name=model_name,
            model_module=model_module,
            csv_path=csv_path
        )
        
        print("\n--- 🎉 Training Finished ---")
        input("Press Enter to return to the model menu...")