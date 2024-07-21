import os
import sys
# sys.path.append("/media/xyw/831bebd9-c866-4ece-b878-5dbd68e5ca50/sjtu/ChengYuLin/CoMER-master")
sys.path.append("/home/bml/storage/mnt/v-5b11ab9b2a534d9b/org/linchengyu/back-tam/COMER_latest/remove3")
import typer
from comer_pos_version1.datamodule import CROHMEDatamodule
from comer_pos_version1.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def main(version: str, test_year: str):
    # generate output latex in result.zip
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size = 1)

    model = LitCoMER.load_from_checkpoint(ckp_path)
    # for p in model.parameters():
    #     print(p)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'Total number of parameters: {total_params}')
    # total_params = 0
    # for name,parameters in model.named_parameters():
    #             # print(name,':',parameters.size())
    #     if name in ['comer_model.posdecoder.pos_embed.0.weight',
    #     'comer_model.posdecoder.pos_embed.0.bias',
    #     'comer_model.posdecoder.pos_embed.2.weight' ,
    #     'comer_model.posdecoder.pos_embed.2.bias',
    #     'comer_model.posdecoder.layernum_proj.0.weight',
    #     'comer_model.posdecoder.layernum_proj.0.bias' ,
    #     'comer_model.posdecoder.layernum_proj.2.weight' ,
    #     'comer_model.posdecoder.layernum_proj.2.bias' ,
    #     'comer_model.posdecoder.pos_proj.0.weight' ,
    #     'comer_model.posdecoder.pos_proj.0.bias' ,
    #     'comer_model.posdecoder.pos_proj.2.weight',
    #     'comer_model.posdecoder.pos_proj.2.bias']:
    #         print(name,parameters.numel())
    #         total_params += parameters.numel() 
    # print(model.parameters())
    # total_params = sum(
    # p.numel() for layer in ['comer_model.posdecoder.pos_embed', 
    #                         'comer_model.posdecoder.layernum_proj', 
    #                         'comer_model.posdecoder.pos_proj'] 
    # for p in layer.parameters()
    # )
    # print(f'Total number of parameters: {total_params}')

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
