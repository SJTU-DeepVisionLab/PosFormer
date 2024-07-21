import os
import sys
sys.path.append("/path/to/PosFormer")
import typer
import zipfile
from Pos_Former.datamodule import CROHMEDatamodule
from Pos_Former.lit_posformer import LitPosFormer
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)

def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range (m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]

def main(version: str, test_year: str):
    # generate output latex in result.zip
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size = 1)

    model = LitPosFormer.load_from_checkpoint(ckp_path)
    trainer.test(model, datamodule=dm)
    caption = {}
    with zipfile.ZipFile("data_crohme.zip") as archive:
        with archive.open(f"data/{test_year}/caption.txt", "r") as f:
            caption_lines = [line.decode('utf-8').strip() for line in f.readlines()]
            for caption_line in caption_lines:
                caption_parts = caption_line.split()
                caption_file_name = caption_parts[0]
                caption_string = ' '.join(caption_parts[1:])
                caption[caption_file_name] = caption_string
    
    with zipfile.ZipFile("result.zip") as archive:
        exprate=[0,0,0,0]
        file_list = archive.namelist()
        txt_files = [file for file in file_list if file.endswith('.txt')]
        for txt_file in txt_files:
            file_name = txt_file.rstrip('.txt')
            with archive.open(txt_file) as f:
                lines = f.readlines()
                pred_string = lines[1].decode('utf-8').strip()[1:-1]
                if file_name in caption:
                    caption_string = caption[file_name]
                else:
                    print(file_name,"not found in caption file")
                    continue
                caption_parts = caption_string.strip().split()
                pred_parts = pred_string.strip().split()
                if caption_string == pred_string:
                    exprate[0]+=1
                else:
                    # print(caption_string)
                    # print(pred_string)
                    error_num=cal_distance(pred_parts,caption_parts)
                    if error_num<=3:
                        exprate[error_num]+=1
        tot = len(txt_files)
        exprate_final=[]
        for i in range(1,5):
            exprate_final.append(100*sum(exprate[:i])/tot)
        print(test_year,"exprate",exprate_final)



if __name__ == "__main__":
    typer.run(main)
