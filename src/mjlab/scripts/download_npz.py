import wandb
run = wandb.init()
artifact = run.use_artifact('zhe92/csv_to_npz/gangnam:v1', type='motions')
artifact_dir = artifact.download()