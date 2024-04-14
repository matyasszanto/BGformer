#run.py testing

from run import main
from pathlib import Path
import os

basepath = Path(os.path.dirname(os.path.abspath(__file__)))
m4_path = Path.joinpath(basepath, "dataset/m4")

if __name__ == "__main__":
      main(["--data_path", "m4-info.csv",
            "--task_name", "short_term_forecast",
            "--is_training", "1",
            "--root_path", f"{m4_path}",
            "--seasonal_patterns", "Daily",
            "--model_id", "m4_Daily",
            "--model", "TimesNet",
            "--data", "m4",
            "--features", "M",
            "--e_layers", "2",
            "--d_layers", "1",
            "--factor", "3",
            "--enc_in", "1",
            "--dec_in", "1",
            "--c_out", "1",
            "--batch_size", "16",
            "--d_model", "16",
            "--d_ff", "16",
            "--top_k", "5",
            "--des", "Exp",
            "--itr", "1",
            "--learning_rate", "0.001",
            "--loss", "SMAPE",
            "--is_training", "0",
            "--seq_len", "28",
            "--label_len", "14",
            "--pred_len", "14",
            ])