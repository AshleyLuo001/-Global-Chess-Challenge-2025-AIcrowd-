import os
import io
import re
import pandas as pd
import polars as pl
import chess
import chess.pgn
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

def load_and_concat_parquets(root_dir: str, file_pattern: str = "*.parquet", path_col: str = "file_path") -> pd.DataFrame:
    root = Path(root_dir)
    dfs = []
    for pq_file in root.rglob(file_pattern):
        try:
            df = pd.read_parquet(pq_file)
            df[path_col] = str(pq_file.resolve())
            dfs.append(df)
            print(f"✔ 读取: {pq_file} （共 {len(df)} 行）")
        except Exception as e:
            print(f"✖️ 无法读取 {pq_file}: {e}")

    if not dfs:
        raise FileNotFoundError(f"未找到任何匹配 '{file_pattern}' 的文件。")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n最终合并得到 {len(dfs)} 个文件，共 {len(combined)} 行。")
    return combined

def fen_to_chess_position(fen: str, legal_moves_uci: str) -> str:
    board_squares = [
        'a1','b1','c1','d1','e1','f1','g1','h1',
        'a2','b2','c2','d2','e2','f2','g2','h2',
        'a3','b3','c3','d3','e3','f3','g3','h3',
        'a4','b4','c4','d4','e4','f4','g4','h4',
        'a5','b5','c5','d5','e5','f5','g5','h5',
        'a6','b6','c6','d6','e6','f6','g6','h6',
        'a7','b7','c7','d7','e7','f7','g7','h7',
        'a8','b8','c8','d8','e8','f8','g8','h8',
    ]
    piece_map = {
        'P': '<White_Pawn>',  'N': '<White_Knight>', 'B': '<White_Bishop>',
        'R': '<White_Rook>',  'Q': '<White_Queen>',  'K': '<White_King>',
        'p': '<Black_Pawn>',  'n': '<Black_Knight>', 'b': '<Black_Bishop>',
        'r': '<Black_Rook>',  'q': '<Black_Queen>',  'k': '<Black_King>',
    }

    fen_parts = fen.split()
    board_fen, turn, castling, en_passant, halfmove, fullmove = fen_parts[:6]
    ranks = list(reversed(board_fen.split('/')))

    board_array = []
    for rank in ranks:
        for ch in rank:
            if ch.isdigit():
                for _ in range(int(ch)):
                    board_array.append('<blank>')
            else:
                board_array.append(piece_map[ch])

    parts = [f"<{board_squares[i]}>{board_array[i]}" for i in range(64)]
    side = "White" if turn == 'w' else "Black"
    parts.append(f"|{side}|{castling}|{en_passant}|{halfmove}|{fullmove}|")

    legal_moves_list = legal_moves_uci.split()
    color = 'White' if turn == 'w' else 'Black'
    encoded_moves = []
    for move in legal_moves_list:
        from_sq, to_sq = move[0:2], move[2:4]
        promotion = move[4:5] if len(move) > 4 else ''
        promo_token = ''
        if promotion:
            promo_token = f"<{color}_{promotion.replace('q','Queen').replace('r','Rook').replace('b','Bishop').replace('n','Knight')}>"
        encoded_moves.append(f"<{from_sq}><{to_sq}>{promo_token}")

    parts.append(" ".join(encoded_moves))
    return "".join(parts)

def extract_best_move_from_comment(comment: str):
    if not comment: return None
    match = re.search(r'([a-zA-Z0-9+#]+)\s+(?:was|is)\s+best', comment)
    return match.group(1) if match else None

def process_single_game_safe(pgn_text: str):
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return []

    if game is None: return []
    board = game.board()
    samples = []

    try:
        for i, node in enumerate(game.mainline()):
            fen = board.fen()
            legal_moves = list(board.legal_moves)
            if not legal_moves: break 
            
            legal_moves_uci = " ".join([m.uci() for m in legal_moves])
            target_move_uci = node.move.uci() 
            is_valid_sample = True
            
            if any(nag in {2, 4, 6} for nag in node.nags):
                corrected_san = extract_best_move_from_comment(node.comment)
                if corrected_san:
                    try:
                        target_move_uci = board.parse_san(corrected_san).uci()
                    except ValueError:
                        is_valid_sample = False
                else:
                    is_valid_sample = False
            
            if is_valid_sample:
                samples.append({
                    "in": fen_to_chess_position(fen, legal_moves_uci),
                    "out": target_move_uci
                })
            
            try:
                board.push(node.move)
            except ValueError:
                break
    except Exception:
        pass
        
    return samples

if __name__ == "__main__":
    version = "v08"
    input_dir = "data/lichess_2025_09"
    output_csv = f"data/feature/train_{version}.csv"
    
    os.makedirs("data/feature", exist_ok=True)
    
    # 1. 加载数据并过滤
    df_all = load_and_concat_parquets(input_dir)
    df_all = df_all[df_all['Result'].isin(['1-0','0-1','1/2-1/2'])].reset_index(drop=True)
    
    # 2. 多进程处理 PGN
    print("Starting parallel PGN parsing...")
    results = Parallel(n_jobs=60)(
        delayed(process_single_game_safe)(pgn) 
        for pgn in tqdm(df_all['movetext'], total=len(df_all))
    )
    
    # 3. 展平与去重处理
    flat_samples = [item for sublist in results for item in sublist]
    df = pl.DataFrame(flat_samples)
    df = df.unique(subset=["in"], keep='first', maintain_order=True)
    df.columns = ['query', 'response']
    
    # 4. 包装 XML tag
    df = df.with_columns(('<chess_position>' + pl.col("query") + '</chess_position>').alias("query"))
    df = df.with_columns(('<uci_move>' + pl.col("response") + '</uci_move>').alias("response"))
    
    # 5. 保存
    df.write_csv(output_csv)
    print(f"Dataset successfully saved to {output_csv}")