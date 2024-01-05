import os
import argparse
import glob
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from TSMixers.data_loader.dataloader_01 import TSFDataLoader
from models.tsmixer import TSMixer, FullLinear, CNN
from callbacks import EarlyStopping, MyPrintingCallback

torch.set_float32_matmul_precision("medium") # to make lightning happy

class TimeSeriesForecasting(pl.LightningModule):
    def __init__(self, args, data_loader):
        super(TimeSeriesForecasting, self).__init__()
        self.args = args
        self.data_loader = data_loader

        if args.model == 'tsmixer':
            self.model = TSMixer(
                input_shape=(args.seq_len, data_loader.n_feature),
                pred_len=args.pred_len,
                norm_type=args.norm_type,
                activation=args.activation,
                dropout=args.dropout,
                n_block=args.n_block,
                ff_dim=args.ff_dim,
                target_slice=data_loader.target_slice
            )
        elif args.model == 'full_linear':
            self.model = FullLinear(
                n_channel=data_loader.n_feature,
                pred_len=args.pred_len
            )
        elif args.model == 'cnn':
            self.model = CNN(
                n_channel=data_loader.n_feature,
                pred_len=args.pred_len,
                kernel_size=args.kernel_size
            )
        else:
            raise ValueError(f'Model not supported: {args.model}')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

def main():
    args = argparse.ArgumentParser(
      description='TSMixer for Time Series Forecasting'
  )

    data_loader = TSFDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target
    )
    
    model = TimeSeriesForecasting(args, data_loader)
    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        progress_bar_refresh_rate=0,
        accelerator=args.ACCELERATOR,
        #strategy=strategy,
        devices=args.DEVICES,
        min_epochs=1,
        max_epochs=args.NUM_EPOCHS,
        precision=args.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    
    start_training_time = time.time()
    trainer.fit(model, train_dataloader=data_loader.train_dataloader())
    trainer.validate(model, val_dataloaders=data_loader.val_dataloader())
    #trainer.fit(model, train_dataloader=data_loader.train_dataloader(), val_dataloaders=data_loader.val_dataloader())
    end_training_time = time.time()
    elapsed_training_time = end_training_time - start_training_time
    print(f'Training finished in {elapsed_training_time} seconds')
    test_result=trainer.test(model, test_dataloader=data_loader.test_dataloader())
    #test_result = trainer.test(test_dataloaders=data_loader.get_test())[0]
    
    # save result to csv
    data = {
        'data': [args.data],
        'model': [args.model],
        'seq_len': [args.seq_len],
        'pred_len': [args.pred_len],
        'lr': [args.learning_rate],
        'mse': [test_result['test_loss']],
        'val_mse': [trainer.callback_metrics['val_loss']],
        'train_mse': [trainer.callback_metrics['train_loss']],
        'training_time': elapsed_training_time,
        'norm_type': args.norm_type,
        'activation': args.activation,
        'n_block': args.n_block,
        'dropout': args.dropout
    }
    if args.model == 'tsmixer':
        data['ff_dim'] = args.ff_dim

    df = pd.DataFrame(data)
    if os.path.exists(args.result_path):
        df.to_csv(args.result_path, mode='a', index=False, header=False)
    else:
        df.to_csv(args.result_path, mode='w', index=False, header=True)

if __name__ == '__main__':
    main()
