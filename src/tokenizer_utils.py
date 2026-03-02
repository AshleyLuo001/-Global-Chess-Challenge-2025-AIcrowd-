import os
import chess
from transformers import AutoTokenizer

def setup_chess_tokenizer(model_name: str, output_dir: str):
    """
    Register custom Special Tokens for chess tasks and save the modified tokenizer.
    
    Args:
        model_name (str): Name or path of the pretrained model to load tokenizer from
        output_dir (str): Directory path to save the customized tokenizer
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 1. Generate all chess square tokens (a1 to h8)
    squares = [f"<{chess.square_name(sq)}>" for sq in chess.SQUARES]
    
    # 2. Generate tokens for all chess pieces and blank position
    pieces = [
        '<White_Pawn>', '<White_Knight>', '<White_Bishop>', 
        '<White_Rook>', '<White_Queen>', '<White_King>',
        '<Black_Pawn>', '<Black_Knight>', '<Black_Bishop>',
        '<Black_Rook>', '<Black_Queen>', '<Black_King>',
        '<blank>'
    ]
    
    special_tokens = {
        'additional_special_tokens': squares + pieces
    }
    
    print(f"Adding {len(squares) + len(pieces)} special tokens...")
    tokenizer.add_special_tokens(special_tokens)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save the customized tokenizer to specified directory
    tokenizer.save_pretrained(output_dir)
    print(f"Custom tokenizer successfully saved to {output_dir}")

if __name__ == "__main__":
    # Default: based on Qwen3-0.6B, save to same-name/custom directory
    setup_chess_tokenizer("Qwen3-0.6B", "models/Qwen3-0.6B-custom")