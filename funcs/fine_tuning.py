from utils.make_save_directory import make_save_directory
import pytorch_lightning as pl
from funcs.auto_model_for_sequence_classification_pl import AutoModelForSequenceClassification_pl

def fine_tuning(dataloader_train, dataloader_val, dataloader_test, num_labels, cfg, tokenizer):
    #save directory作成
    model_save_dir = make_save_directory(cfg)

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath=model_save_dir,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=3,
        mode='min',
    )
    callbacks = [checkpoint, lr_monitor, early_stopping]
    mlf_logger = pl.loggers.MLFlowLogger(experiment_name=cfg.model.exp_name,
                                         tracking_uri='file' + cfg.directory.set_tracking_uri_dir +'/mlruns')
    
    #mlflow log param
    for i in ['model', 'hyper_parameter']:
        for k, v in cfg[i].items():
            mlf_logger.log_hyperparams({i + '.' + k: v})
    

    trainer = pl.Trainer(
        #gpu=cfg.hyper_parameter.gpus,
        devices=cfg.hyper_parameter.gpus,
        max_epochs=cfg.hyper_parameter.max_epochs,
        callbacks=callbacks,
        logger=mlf_logger,
    )

    model = AutoModelForSequenceClassification_pl(
        model_name=cfg.model.model_name,
        num_labels=num_labels,
        lr=cfg.model.lr,
        tokenizer=tokenizer
    )

    #Fine-Tuning
    trainer.fit(
        model,
        dataloader_train,
        dataloader_val
    )

    best_model_path = checkpoint.best_model_path

    test = trainer.test(dataloaders=dataloader_test)
    mlf_logger.log_metrics({'Accurasy':test[0]['accuracy']})

    #Fine-Tuningモデルの保存
    model = AutoModelForSequenceClassification_pl.load_from_checkpoint(best_model_path)
    model.bert.save_pretrained(model_save_dir)